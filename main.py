from googleapiclient.discovery import build
from datetime import datetime
import pandas as pd
import os
import argparse
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
import openai
import sqlite3
import uuid
import logging
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging(log_level=logging.INFO):
    """Configure logging with both file and console handlers"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create rotating file handler
    file_handler = RotatingFileHandler(
        'logs/youtube_transcript.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Default values
DEFAULT_API_KEY = os.getenv('YOUTUBE_API_KEY')
DEFAULT_OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEFAULT_PLAYLIST_ID = 'PLU9-uwewPMe2ACTcry7ChkTbujexZnjlN'  # 조코딩
DEFAULT_DB_PATH = 'youtube_data.db'
DEFAULT_CSV_PATH = f'youtube_data_{datetime.now().strftime("%Y%m%d")}.csv'

# Create logger
logger = setup_logging()

def create_database(db_path):
    """Create SQLite database and necessary tables"""
    logger.info(f"Creating/connecting to database at {db_path}")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            uuid TEXT PRIMARY KEY,
            video_id TEXT UNIQUE,
            video_url TEXT,
            title TEXT,
            description TEXT,
            published_at TIMESTAMP,
            transcript_ko TEXT,
            transcript_en TEXT,
            summary_ko TEXT,
            summary_en TEXT
        )
    ''')
    
    conn.commit()
    logger.debug("Database schema created successfully")
    return conn

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract YouTube playlist video information')
    parser.add_argument('--youtube-api-key', '-ytk', 
                      default=DEFAULT_API_KEY,
                      help='YouTube Data API key (default: uses preset API key)')
    parser.add_argument('--playlist-id', '-p',
                      default=DEFAULT_PLAYLIST_ID,
                      help='YouTube playlist ID (default: JoCoding playlist)')
    parser.add_argument('--output-csv', '-o',
                      default=DEFAULT_CSV_PATH,
                      help='Output CSV filename (default: youtube_data_YYYYMMDD.csv)')
    parser.add_argument('--db-path', '-d',
                      default=DEFAULT_DB_PATH,
                      help='SQLite database path (default: youtube_data.db)')
    parser.add_argument('--openai-api-key', '-oaik',
                      default=DEFAULT_OPENAI_API_KEY,
                      help='OpenAI API key for generating summaries')
    return parser.parse_args()

def get_video_transcript(video_id):
    """
    Fetch transcript for a YouTube video in both Korean and English.
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        tuple: (korean_transcript, english_transcript) or (None, None) if not available
    """
    logger.info(f"Fetching transcripts for video ID: {video_id}")
    try:
        # Try Korean transcript
        ko_transcript = None
        try:
            ko_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
            ko_transcript = ' '.join([item['text'] for item in ko_list])
            logger.debug(f"Korean transcript fetched successfully for {video_id}")
        except (TranscriptsDisabled, NoTranscriptFound):
            logger.warning(f"No Korean transcript available for {video_id}")
            pass

        # Try English transcript
        en_transcript = None
        try:
            en_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            en_transcript = ' '.join([item['text'] for item in en_list])
            logger.debug(f"English transcript fetched successfully for {video_id}")
        except (TranscriptsDisabled, NoTranscriptFound):
            logger.warning(f"No English transcript available for {video_id}")
            pass
        
        return ko_transcript, en_transcript
    
    except Exception as e:
        logger.error(f"Error fetching transcript for {video_id}: {str(e)}", exc_info=True)
        return None, None

def get_playlist_videos(youtube, playlist_id):
    videos = []
    
    next_page_token = None
    while True:
        request = youtube.playlistItems().list(
            part='snippet,contentDetails',  # Add contentDetails to get full content
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response['items']:
            video_id = item['snippet']['resourceId']['videoId']
            
            # Get full video details
            video_response = youtube.videos().list(
                part='snippet,contentDetails',
                id=video_id
            ).execute()
            
            if video_response['items']:
                video_details = video_response['items'][0]
                # Get complete description without truncation
                full_description = video_details['snippet']['description']
                
                video = {
                    'uuid': str(uuid.uuid4()),
                    'video_id': video_id,
                    'video_url': f'https://www.youtube.com/watch?v={video_id}',
                    'title': video_details['snippet']['title'],
                    'description': full_description,
                    'published_at': video_details['snippet']['publishedAt']
                }
                videos.append(video)
        
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    
    return videos

def estimate_tokens(text):
    """
    Estimate the number of tokens in a text.
    This is a rough estimation: ~4 characters per token for English, ~2-3 for Korean.
    """
    # Count Korean characters (more tokens per character)
    korean_char_count = sum(1 for char in text if '\u3130' <= char <= '\u318F' or '\uAC00' <= char <= '\uD7AF')
    # Count other characters
    other_char_count = len(text) - korean_char_count
    
    # Estimate tokens: Korean chars * 0.5 (2 chars per token) + other chars * 0.25 (4 chars per token)
    estimated_tokens = int(korean_char_count * 0.5 + other_char_count * 0.25)
    return estimated_tokens

def chunk_text(text, target_token_size=1000):
    """
    Split text into chunks while preserving sentence boundaries and respecting token limits.
    
    Args:
        text (str): Text to split
        target_token_size (int): Target size of each chunk in tokens
    
    Returns:
        list: List of text chunks
    """
    # Split text into sentences
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip() + '. '
        sentence_tokens = estimate_tokens(sentence)
        
        # If a single sentence exceeds target size, split it into smaller parts
        if sentence_tokens > target_token_size:
            # Split long sentence by commas first
            comma_parts = sentence.split(', ')
            for part in comma_parts:
                part = part.strip() + ', '
                part_tokens = estimate_tokens(part)
                
                if current_tokens + part_tokens > target_token_size and current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = [part]
                    current_tokens = part_tokens
                else:
                    current_chunk.append(part)
                    current_tokens += part_tokens
        else:
            if current_tokens + sentence_tokens > target_token_size and current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return chunks

def generate_bullet_summary(transcript_text, target_language="ko"):
    """
    Generate a bullet-point formatted summary of transcript text using OpenAI's API.
    Handles long transcripts by chunking and iterative summarization.
    
    Args:
        transcript_text (str): The transcript text to summarize
        target_language (str): Target language for the summary (ko or en)
    
    Returns:
        str: The generated bullet-point summary
        None: If an error occurs
    """
    if not transcript_text:
        logger.warning("No transcript text provided for summarization")
        return None
    
    try:
        # Initialize the OpenAI client
        client = openai.OpenAI()
        
        # Language-specific instructions
        lang_instruction = "한글로" if target_language == "ko" else "in English"
        
        # Step 1: Split long text into chunks (target ~1000 tokens per chunk)
        chunks = chunk_text(transcript_text, target_token_size=1000)
        chunk_summaries = []
        
        # Step 2: Generate initial summaries for each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            prompt = f"""Please provide a concise bullet-point summary of this transcript section {lang_instruction}. 
Use hierarchical bullet points (•, -, ◦) to show the structure and relationships between ideas.
Focus on key points and maintain the logical flow.

Transcript section {i+1}/{len(chunks)}:
{chunk}"""
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates concise, well-structured bullet-point summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500  # Reduced to prevent overflow
                )
                chunk_summaries.append(response.choices[0].message.content.strip())
                logger.debug(f"Successfully generated summary for chunk {i+1}")
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}", exc_info=True)
                # Continue with next chunk if one fails
                continue
        
        # Step 3: Combine summaries in batches if needed
        while len(chunk_summaries) > 1:
            logger.info(f"Combining {len(chunk_summaries)} summaries")
            new_summaries = []
            
            # Process summaries in pairs
            for i in range(0, len(chunk_summaries), 2):
                if i + 1 < len(chunk_summaries):
                    # Combine pair of summaries
                    pair = chunk_summaries[i:i+2]
                    combined_text = "\n\n".join(pair)
                    
                    try:
                        final_prompt = f"""Please create a cohesive, well-structured bullet-point summary {lang_instruction} by combining these section summaries.
Maintain the hierarchical structure (•, -, ◦) and ensure logical flow.
Eliminate redundancy while preserving important information.

Summaries to combine:
{combined_text}"""
                        
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that combines summaries while maintaining structure and clarity."},
                                {"role": "user", "content": final_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=800
                        )
                        new_summaries.append(response.choices[0].message.content.strip())
                        logger.debug(f"Successfully combined summaries {i} and {i+1}")
                    except Exception as e:
                        logger.error(f"Error combining summaries: {str(e)}", exc_info=True)
                        # If error occurs, keep original summaries
                        new_summaries.extend(pair)
                else:
                    # Add remaining summary if odd number
                    new_summaries.append(chunk_summaries[i])
            
            # Update summaries for next iteration
            chunk_summaries = new_summaries
        
        return chunk_summaries[0] if chunk_summaries else None
    
    except Exception as e:
        logger.error(f"Error in summary generation: {str(e)}", exc_info=True)
        return None

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("Starting YouTube transcript extraction and summarization")
    
    # Create YouTube API client
    youtube = build('youtube', 'v3', developerKey=args.youtube_api_key)
    
    # Set OpenAI API key
    if args.openai_api_key:
        openai.api_key = args.openai_api_key
        logger.debug("OpenAI API key set successfully")
    
    # Create database connection
    conn = create_database(args.db_path)
    cursor = conn.cursor()
    
    # Get videos from playlist
    logger.info(f"Fetching videos from playlist: {args.playlist_id}")
    videos = get_playlist_videos(youtube, args.playlist_id)
    total_videos = len(videos)
    logger.info(f"Found {total_videos} videos in playlist")
    
    # Process each video
    for idx, video in enumerate(videos, 1):
        logger.info(f"Processing video [{idx}/{total_videos}]: {video['title']}")
        
        # Get transcripts
        ko_transcript, en_transcript = get_video_transcript(video['video_id'])
        
        # Generate summaries if transcripts are available
        ko_summary = None
        en_summary = None
        
        if ko_transcript:
            logger.info(f"[{idx}/{total_videos}] Generating Korean summary...")
            ko_summary = generate_bullet_summary(ko_transcript, "ko")
        
        if en_transcript:
            logger.info(f"[{idx}/{total_videos}] Generating English summary...")
            en_summary = generate_bullet_summary(en_transcript, "en")
        
        # Store in database
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO videos 
                (uuid, video_id, video_url, title, description, published_at, 
                 transcript_ko, transcript_en, summary_ko, summary_en)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                video['uuid'],
                video['video_id'],
                video['video_url'],
                video['title'],
                video['description'],
                video['published_at'],
                ko_transcript,
                en_transcript,
                ko_summary,
                en_summary
            ))
            conn.commit()
            logger.debug(f"[{idx}/{total_videos}] Successfully saved data for video {video['video_id']}")
        except Exception as e:
            logger.error(f"[{idx}/{total_videos}] Error saving video data to database: {str(e)}", exc_info=True)
    
    # Create DataFrame from database for CSV export
    try:
        df = pd.read_sql_query('''
            SELECT * FROM videos 
            ORDER BY published_at DESC
        ''', conn)
        
        # Save to CSV
        df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
        logger.info(f"Data saved to database: {args.db_path}")
        logger.info(f"Data exported to CSV: {args.output_csv}")
    except Exception as e:
        logger.error(f"Error exporting data to CSV: {str(e)}", exc_info=True)
    
    # Close database connection
    conn.close()
    logger.info(f"Processing completed. Processed {total_videos} videos in total.")

if __name__ == "__main__":
    main()
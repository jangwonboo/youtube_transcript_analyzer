# YouTube Transcript Analyzer v1.1

A powerful tool for analyzing YouTube video transcripts with enhanced summarization capabilities using OpenAI GPT and SQLite3 storage.

## Features

- Fetch transcripts from YouTube videos in both Korean and English
- Generate hierarchical bullet-point summaries using OpenAI GPT
- Store data in SQLite3 database for improved stability
- Export data to CSV format
- Comprehensive logging system with progress tracking
- Support for processing entire playlists
- Token-aware chunking for handling long transcripts

## New in Version 1.1

- Added SQLite3 database support for improved data stability
- Enhanced summary feature with token-aware chunking
- Improved error handling and logging system
- Progress tracking for long-running processes
- Support for both Korean and English transcripts and summaries

## Requirements

- Python 3.8+
- OpenAI API key
- YouTube Data API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jangwonboo/youtube_transcript_analyzer.git
cd youtube_transcript_analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
YOUTUBE_API_KEY=your_youtube_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

Basic usage:
```bash
python main.py --playlist-id PLAYLIST_ID
```

All options:
```bash
python main.py --youtube-api-key YOUR_YT_KEY \
               --openai-api-key YOUR_OPENAI_KEY \
               --playlist-id PLAYLIST_ID \
               --output-csv output.csv \
               --db-path data.db
```

## Output

The script generates:
1. SQLite database with all video data
2. CSV export file
3. Detailed logs in the `logs` directory

## Data Structure

The following data is stored for each video:
- UUID (unique identifier)
- Video ID
- Video URL
- Title
- Description
- Published Date
- Korean Transcript
- English Transcript
- Korean Summary (bullet-point format)
- English Summary (bullet-point format)

## License

MIT License

## Author

Jangwon Boo
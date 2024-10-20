import gradio as gr
from transformers import pipeline, AutoTokenizer
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# YouTube Data API setup
API_KEY = "Youtube_Data_Api_Key"
youtube = build("youtube", "v3", developerKey=API_KEY)

def analyze_transcript_sentiment(video_id):
    """
    Fetches the transcript and performs sentiment analysis on each line.
    """
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    transcript_lines = []
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([entry['text'] for entry in transcript])

        lines = transcript_text.splitlines()

        for line in lines:
            if len(tokenizer.encode(line)) <= 512:
                sentiment = sentiment_analyzer(line)[0]
                transcript_lines.append({
                    "line": line,
                    "label": sentiment["label"],
                    "score": sentiment["score"]
                })

                if sentiment["label"] == "POSITIVE":
                    sentiments["positive"] += 1
                elif sentiment["label"] == "NEGATIVE":
                    sentiments["negative"] += 1
                else:
                    sentiments["neutral"] += 1

        total_sentiments = sum(sentiments.values())
        sentiments["positive_percentage"] = (sentiments["positive"] / total_sentiments) * 100
        sentiments["negative_percentage"] = (sentiments["negative"] / total_sentiments) * 100
        sentiments["neutral_percentage"] = (sentiments["neutral"] / total_sentiments) * 100

        return transcript_lines, sentiments
    except Exception as e:
        return f"Error: {e}"

def fetch_youtube_comments(video_id):
    """
    Fetches the top comments for a YouTube video.
    """
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=50,  
            textFormat="plainText"
        )
        response = request.execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        return comments
    except Exception as e:
        return f"Error fetching comments: {e}"

def analyze_comments_sentiment(video_id):
    """
    Fetches comments and performs sentiment analysis on each.
    """
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    comments = fetch_youtube_comments(video_id)
    if isinstance(comments, str):
        return comments, sentiments  

    comment_sentiments = []
    for comment in comments:
        sentiment = sentiment_analyzer(comment)[0]
        comment_sentiments.append({
            "comment": comment,
            "label": sentiment["label"],
            "score": sentiment["score"]
        })

        if sentiment["label"] == "POSITIVE":
            sentiments["positive"] += 1
        elif sentiment["label"] == "NEGATIVE":
            sentiments["negative"] += 1
        else:
            sentiments["neutral"] += 1

    total_sentiments = sum(sentiments.values())
    sentiments["positive_percentage"] = (sentiments["positive"] / total_sentiments) * 100
    sentiments["negative_percentage"] = (sentiments["negative"] / total_sentiments) * 100
    sentiments["neutral_percentage"] = (sentiments["neutral"] / total_sentiments) * 100
    
    return comment_sentiments, sentiments


def transcript_analysis(video_id):
    transcript_results, sentiment_counts = analyze_transcript_sentiment(video_id)
    if isinstance(transcript_results, str):
        return transcript_results  
    
    output = "Transcript Sentiment Analysis:\n"
    for result in transcript_results:
        output += f"Line: {result['line']}\nSentiment: {result['label']} (Score: {result['score']:.2f})\n\n"

    output += f"Sentiment Summary: Positive: {sentiment_counts['positive']} ({sentiment_counts['positive_percentage']:.2f}%), Negative: {sentiment_counts['negative']} ({sentiment_counts['negative_percentage']:.2f}%), Neutral: {sentiment_counts['neutral']} ({sentiment_counts['neutral_percentage']:.2f}%)\n"

    return output.strip()

def comment_analysis(video_id):
    comment_results, sentiment_counts = analyze_comments_sentiment(video_id)
    if isinstance(comment_results, str):
        return comment_results  
    
    output = "Comments Sentiment Analysis:\n"
    for result in comment_results:
        output += f"Comment: {result['comment']}\nSentiment: {result['label']} (Score: {result['score']:.2f})\n\n"

    output += f"Sentiment Summary: Positive: {sentiment_counts['positive']} ({sentiment_counts['positive_percentage']:.2f}%), Negative: {sentiment_counts['negative']} ({sentiment_counts['negative_percentage']:.2f}%), Neutral: {sentiment_counts['neutral']} ({sentiment_counts['neutral_percentage']:.2f}%)\n"

    return output.strip()

# Gradio Interface

with gr.Blocks() as demo:
    gr.Markdown("# YouTube Video Sentiment Analysis")
    gr.Markdown("## Analyze Transcript and Comments Separately")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Transcript Analysis")
            transcript_video_id = gr.Textbox(label="YouTube Video ID for Transcript", placeholder="Enter YouTube video ID")
            transcript_output = gr.Textbox(label="Transcript Sentiment Analysis Output", lines=10, elem_id="transcript_output")
            transcript_button = gr.Button("Analyze Transcript", elem_id="transcript_button")
        
        with gr.Column():
            gr.Markdown("### Comment Analysis")
            comment_video_id = gr.Textbox(label="YouTube Video ID for Comments", placeholder="Enter YouTube video ID")
            comment_output = gr.Textbox(label="Comments Sentiment Analysis Output", lines=10, elem_id="comment_output")
            comment_button = gr.Button("Analyze Comments", elem_id="comment_button")

    gr.HTML("""
    <style>
        #transcript_button {
            background-color: #FF8C00;  /* Orange */
            color: white; 
            font-weight: bold;
        }
        #comment_button {
            background-color: #4CAF50; /* Green */
            color: white; 
            font-weight: bold;
        }
        .gr-box {
            border-color: #FFD700;  /* Gold border for the box */
        }
        #transcript_output, #comment_output {
            border-color: #FFD700;
        }
    </style>
    """)
    
    transcript_button.click(transcript_analysis, inputs=transcript_video_id, outputs=transcript_output)
    comment_button.click(comment_analysis, inputs=comment_video_id, outputs=comment_output)

demo.launch()
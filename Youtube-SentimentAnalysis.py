import dash
from dash import dcc, html, Input, Output, State
from transformers import pipeline, AutoTokenizer
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])  

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
API_KEY = "API_KEY"
youtube = build("youtube", "v3", developerKey=API_KEY)

def analyze_transcript_sentiment(video_id):
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
        if "Could not retrieve a transcript" in str(e):
            return "Transcript not available for this video.", None 
        else:
            return f"Error: {e}", None  

def analyze_comments_sentiment(video_id):
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
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

    except Exception as e:
        return f"Error fetching comments: {e}"

app.layout = dbc.Container([
    html.H1("YouTube Video Sentiment Analysis", className="text-center mb-5"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H3("Transcript Analysis", className="mb-3"),
            dcc.Input(id="transcript_video_id", type="text", placeholder="Enter YouTube video ID", className="form-control mb-3"),
            dbc.Button("Analyze Transcript", id="transcript_button", color="warning", className="w-100 mb-3", style={"background-color": "#ff7f00", "border-color": "#ff7f00"}),
            dcc.Loading(  
                id="loading_transcript",
                type="dot",  
                children=dbc.Card([
                    dbc.CardBody([
                        html.Div(id="transcript_output", style={"font-size": "16px", "line-height": "1.6", "overflowY": "auto", "maxHeight": "400px", "padding": "15px"})
                    ])
                ])
            )
        ], width=6),
        dbc.Col([
            html.H3("Comment Analysis", className="mb-3"),
            dcc.Input(id="comment_video_id", type="text", placeholder="Enter YouTube video ID", className="form-control mb-3"),
            dbc.Button("Analyze Comments", id="comment_button", color="success", className="w-100 mb-3", style={"background-color": "#ff7f00", "border-color": "#ff7f00"}),
            dcc.Loading(  
                id="loading_comments",
                type="dot",
                children=dbc.Card([
                    dbc.CardBody([
                        html.Div(id="comment_output", style={"font-size": "16px", "line-height": "1.6", "overflowY": "auto", "maxHeight": "400px", "padding": "15px"})
                    ])
                ])
            )
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("Overall Sentiment Summary", className="mb-3"),
                    html.Div(id="overall_sentiment_output", style={"font-size": "16px", "line-height": "1.6", "padding": "15px", "paddingTop": "20px"})
                ])
            ], className="mt-4")
        ], width=12)
    ])
], fluid=True, style={"paddingTop": "50px", "paddingLeft": "15px", "paddingRight": "15px", "paddingBottom": "50px"})

@app.callback(
    Output("transcript_output", "children"),
    Input("transcript_button", "n_clicks"),
    State("transcript_video_id", "value"),
    prevent_initial_call=True
)
def update_transcript_output(n_clicks, video_id):
    if not video_id:
        return "Please enter a valid YouTube video ID."

    transcript_results, sentiment_counts = analyze_transcript_sentiment(video_id)
    if isinstance(transcript_results, str):
        return transcript_results

    output = [html.H4("Transcript Sentiment Analysis:")]
    for result in transcript_results:
        color = (
            "green" if result["label"] == "POSITIVE" 
            else "red" if result["label"] == "NEGATIVE" 
            else "yellow"
        )
        output.append(html.P([ 
            html.Span(f"Line: {result['line']}"), 
            html.Br(),
            html.Span(f"Sentiment: ", style={"font-weight": "bold"}), 
            html.Span(result["label"], style={"color": color}),
            html.Span(f" (Score: {result['score']:.2f})") 
        ]))

    output.append(html.H4("Sentiment Summary:"))
    output.append(html.P(f"Positive: {sentiment_counts['positive']} ({sentiment_counts['positive_percentage']:.2f}%)"))
    output.append(html.P(f"Negative: {sentiment_counts['negative']} ({sentiment_counts['negative_percentage']:.2f}%)"))
    output.append(html.P(f"Neutral: {sentiment_counts['neutral']} ({sentiment_counts['neutral_percentage']:.2f}%)"))

    return output

@app.callback(
    Output("comment_output", "children"),
    Input("comment_button", "n_clicks"),
    State("comment_video_id", "value"),
    prevent_initial_call=True
)
def update_comment_output(n_clicks, video_id):
    if not video_id:
        return "Please enter a valid YouTube video ID."

    comment_results, sentiment_counts = analyze_comments_sentiment(video_id)
    if isinstance(comment_results, str):
        return comment_results

    output = [html.H4("Comments Sentiment Analysis:")]
    for result in comment_results:
        color = (
            "green" if result["label"] == "POSITIVE" 
            else "red" if result["label"] == "NEGATIVE" 
            else "yellow"
        )
        output.append(html.P([ 
            html.Span(f"Comment: {result['comment']}"),
            html.Br(),
            html.Span(f"Sentiment: ", style={"font-weight": "bold"}), 
            html.Span(result["label"], style={"color": color}),
            html.Span(f" (Score: {result['score']:.2f})") 
        ]))

    output.append(html.H4("Sentiment Summary:"))
    output.append(html.P(f"Positive: {sentiment_counts['positive']} ({sentiment_counts['positive_percentage']:.2f}%)"))
    output.append(html.P(f"Negative: {sentiment_counts['negative']} ({sentiment_counts['negative_percentage']:.2f}%)"))
    output.append(html.P(f"Neutral: {sentiment_counts['neutral']} ({sentiment_counts['neutral_percentage']:.2f}%)"))

    return output

@app.callback(
    Output("overall_sentiment_output", "children"),
    Input("transcript_button", "n_clicks"),
    Input("comment_button", "n_clicks"),
    State("transcript_video_id", "value"),
    State("comment_video_id", "value"),
    prevent_initial_call=True
)
def update_overall_sentiment(transcript_n_clicks, comment_n_clicks, transcript_video_id, comment_video_id):
    if not transcript_video_id or not comment_video_id:
        return "Please analyze both transcript and comments first."

    transcript_results, transcript_sentiments = analyze_transcript_sentiment(transcript_video_id)
    comment_results, comment_sentiments = analyze_comments_sentiment(comment_video_id)
    
    if transcript_sentiments is None or comment_sentiments is None:
        return "Error: Could not analyze video."

    overall_sentiment = "Neutral"
    if transcript_sentiments["positive_percentage"] > comment_sentiments["positive_percentage"]:
        overall_sentiment = "Positive" if transcript_sentiments["positive_percentage"] > 50 else "Neutral"
    elif comment_sentiments["positive_percentage"] > 50:
        overall_sentiment = "Positive"
    else:
        overall_sentiment = "Negative"
        
    return html.Span([
        "The overall sentiment of the video is: ", 
        html.B(overall_sentiment)
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
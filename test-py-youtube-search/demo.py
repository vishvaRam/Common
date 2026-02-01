import asyncio
from py_youtube_search import YouTubeSearch, Filters

# Initialize (No API Key required)
yt = YouTubeSearch()

# Search for long tutorials (>20 mins) uploaded this week
results = await yt.search(
    "LangGraph RAG", 
    sp=Filters.long_this_week, 
    limit=2
)

for v in results:
    print(f"🎥 {v['title']}")
    print(f"⏱️ {v['duration']} | 🔗 https://youtu.be/{v['id']}")

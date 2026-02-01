import asyncio
from py_youtube_search import YouTubeSearch, Filters

async def main():
    yt = YouTubeSearch()

    # Search 1: Long videos about LangGraph
    print("Searching for LangGraph...")
    videos = await yt.search("LangGraph", sp=Filters.long_this_week, limit=3)

    for v in videos:
        print(f"🎥 {v['title']} | ⏱ {v['duration']} | 👁 {v['views']}")

    # Search 2: Reusing the same client for a different query
    print("\nSearching for Python...")
    videos_py = await yt.search("Python 3.12", sp=Filters.medium_today, limit=3)
    
    for v in videos_py:
        print(f"🐍 {v['title']}")

if __name__ == "__main__":
    asyncio.run(main())
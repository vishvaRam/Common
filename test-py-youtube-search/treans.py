import asyncio
from youtube_transcript_api import YouTubeTranscriptApi
from py_youtube_search import YouTubeSearch, Filters


async def main():
    yt = YouTubeSearch()
    query = "LangGraph tutorial"

    print(f"🔍 Searching for: '{query}'...")

    videos = await yt.search(query, sp=Filters.long_this_week, limit=1)

    if not videos:
        print("❌ No videos found.")
        return

    video = videos[0]
    vid_id = video["id"]

    print(f"▶ Found: {video['title']} (ID: {vid_id})")
    print("\n📝 Fetching transcript (via youtube-transcript-api v1.2.3)...")

    try:
        loop = asyncio.get_event_loop()

        # Create API instance (THIS IS REQUIRED)
        ytt_api = YouTubeTranscriptApi()

        # fetch() is an instance method
        fetched_transcript = await loop.run_in_executor(
            None,
            lambda: ytt_api.fetch(vid_id)
        )

        # fetched_transcript is a FetchedTranscript object
        # Convert to raw list of dicts if you want
        transcript = fetched_transcript.to_raw_data()

        print(f"\n✅ Transcript Fetched ({len(transcript)} lines)")
        print("-" * 60)
        for line in transcript[:10]:
            print(f"[{line['start']:6.2f}s] {line['text']}")
        print("-" * 60)

    except Exception as e:
        print(f"\n❌ Error fetching transcript: {e}")


if __name__ == "__main__":
    asyncio.run(main())

import re
import urllib.request

class YouTubeSearch:
    def __init__(self, keywords: str, sp: str = None, limit: int = 15):
        self.keywords = keywords.replace(" ", "+")
        self.sp = sp
        self.limit = limit
        self.source = self._fetch_source()

    def _fetch_source(self):
        base_url = f"https://www.youtube.com/results?search_query={self.keywords}"
        if self.sp:
            url = f"{base_url}&sp={self.sp}"
        else:
            url = base_url
            
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            return response.read().decode("utf-8")

    def videos(self):
        # Regex to capture distinct JSON fields for ID, Title, Duration, and Views.
        # This matches the "videoRenderer" block and extracts specific "simpleText" values.
        # Note: Live streams may be skipped as they often lack 'lengthText'.
        pattern = (
            r'\"videoRenderer\":\{'
            r'.+?\"videoId\":\"(?P<id>\S{11})\"'
            r'.+?\"title\":\{\"runs\":\[\{\"text\":\"(?P<title>.+?)\"\}\]'
            r'.+?\"lengthText\":\{.*?\"simpleText\":\"(?P<duration>.+?)\"\}'
            r'.+?\"viewCountText\":\{\"simpleText\":\"(?P<views>.+?)\"\}'
        )

        matches = re.finditer(pattern, self.source)
        
        results = []
        for match in matches:
            if len(results) >= self.limit:
                break
            
            data = match.groupdict()
            results.append({
                "id": data["id"],
                "title": data["title"],
                "duration": data["duration"],  # e.g., "21:05"
                "views": data["views"]         # e.g., "1.2K views"
            })

        return results


# --- Usage ---

FILTERS = {
    # Duration: 3 - 20 Minutes (Medium)
    "medium_today":      "EgYIAhABGAU=",   
    "medium_this_week":  "EgQIAxGF",       
    "medium_this_month": "EgYIBBABGAU=",   
    "medium_this_year":  "EgYIBRABGAU=",   

    # Duration: Over 20 Minutes (Long)
    "long_today":        "EgYIAhABGAI=",   
    "long_this_week":    "EgQIAxAB",       
    "long_this_month":   "EgYIBBABGAI=",   
    "long_this_year":    "EgYIBRABGAI="    
}

# Example: Search for 'LangGraph' using the 'Over 20 mins - This Week' filter
print("--- Searching for Medium Videos (3-20m) from This Year ---")
search = YouTubeSearch("LangGraph", sp=FILTERS["medium_this_year"], limit=5)
videos = search.videos()

print(f"\n🔍 Found {len(videos)} videos:\n")

for i, v in enumerate(videos, 1):
    print(f"{i:02d}. 🎥 {v['title']}")
    print(f"    🆔 ID      : {v['id']}")
    print(f"    ⏱ Duration: {v['duration']}")
    print(f"    👁 Views   : {v['views']}")
    print("-" * 60)

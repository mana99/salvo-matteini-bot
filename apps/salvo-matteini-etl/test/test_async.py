
import asyncio
import aiohttp
from oauth2_auth import access_token
BASE_URL = 'https://api.twitter.com/'


async def query(aiohttp):
    search_headers = {
        'Authorization': 'Bearer {}'.format(access_token)
    }
    search_url = BASE_URL + '1.1/search/tweets.json'
    search_params = {
        'q': 'General Election',
        'result_type': 'recent',
        'count': 2
    }
    async with aiohttp.get(search_url, headers=search_headers, params=search_params) as resp:
        j = await resp.json()

    return j

async def main():
    async with aiohttp.ClientSession() as client:
        res = await query(client)
    return res

# loop = asyncio.get_event_loop()
# client = aiohttp.ClientSession(loop=loop)

res = asyncio.run(main())
print(res)
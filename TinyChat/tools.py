"""
tools.py - what the bot can do besides talk.

The brain in tinychat.py answers some messages with a command in square
brackets, such as [weather], because it learned that those messages need
something it does not have: a clock or the internet. use_tools() spots the
command, does the real work here, and returns a sentence to show the person.

The brain decides WHICH tool to use. Plain code here works out the details
(which city, which subject) and fetches the answer. Everything uses Python's
standard library, and none of the services need an account or a key:

    [time] [date]   your computer's clock
    [weather]       open-meteo.com
    [lookup]        wikipedia.org
    [news]          NPR's public news feed
"""
import datetime
import difflib
import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

USER_AGENT = "tinychat/1.0 (a from-scratch learning chatbot)"
TIMEOUT = 8          # seconds to wait for a website before giving up
HEADLINES = 5        # how many news headlines to show

# Which clock [time] and [date] use. A hosting company's computers usually
# run on a different clock from yours, so name your time zone here. Set this
# to None to use the clock of whatever computer the bot is running on.
TIMEZONE = "America/New_York"


# ----------------------------------------------------------------------------
# Getting things from the internet
# ----------------------------------------------------------------------------
def fetch(url):
    """Download a web address and return its contents as text."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.read().decode("utf-8")
    except urllib.error.URLError as error:
        # Some Python installs on Macs cannot verify secure sites until the
        # "certifi" package is installed. If it is there, try again with it.
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
        try:
            import certifi
        except ImportError:
            raise RuntimeError("certificates") from error
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=TIMEOUT, context=context) as response:
            return response.read().decode("utf-8")


def fetch_json(url, **parameters):
    """Download a web address that answers in JSON, a format for data."""
    return json.loads(fetch(url + "?" + urllib.parse.urlencode(parameters)))


def tidy(message):
    """Lowercase a message and drop the punctuation at its end."""
    return message.lower().strip().rstrip("?!. ").strip()


# ----------------------------------------------------------------------------
# [time] and [date]
# ----------------------------------------------------------------------------
def right_now():
    """The current time in TIMEZONE, or by this computer's clock if unset."""
    if TIMEZONE:
        try:
            from zoneinfo import ZoneInfo
            return datetime.datetime.now(ZoneInfo(TIMEZONE))
        except Exception:
            pass    # unknown zone name: fall back to the computer's clock
    return datetime.datetime.now()


def tell_time(message):
    now = right_now()
    return f"it is {now.strftime('%I:%M %p').lstrip('0').lower()}."


def tell_date(message):
    now = right_now()
    return f"today is {now:%A}, {now:%B} {now.day}, {now.year}.".lower()


# ----------------------------------------------------------------------------
# [weather]
# ----------------------------------------------------------------------------
SKY = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
    51: "drizzling lightly", 53: "drizzling", 55: "drizzling heavily",
    56: "drizzling and freezing", 57: "drizzling and freezing",
    61: "raining lightly", 63: "raining", 65: "raining heavily",
    66: "raining and freezing", 67: "raining and freezing",
    71: "snowing lightly", 73: "snowing", 75: "snowing heavily", 77: "snowing",
    80: "showery", 81: "showery", 82: "pouring", 85: "snowing on and off", 86: "snowing heavily on and off",
    95: "stormy with thunder", 96: "stormy with thunder and hail", 99: "stormy with thunder and hail",
}

# Words in a weather question that are not part of the place name.
NOT_A_PLACE = set("""weather forecast temperature the what whats what's is how it its today tonight now
right like outside please tell me will rain raining sunny cold hot warm snowing snow do does i need a an
umbrella jacket coat and or going to be there here currently current tomorrow this week""".split())


def find_place(message):
    """Pull the place name out of a weather question."""
    text = tidy(message)
    after_in = re.search(r"\b(?:in|for|at|near)\s+(.+)$", text)
    words = (after_in.group(1) if after_in else text).split()
    return " ".join(word for word in words if word not in NOT_A_PLACE)


def weather(message):
    place = find_place(message)
    if not place:
        return "which place? ask me like this: weather in tokyo"

    found = fetch_json("https://geocoding-api.open-meteo.com/v1/search",
                       name=place, count=1, language="en", format="json").get("results")
    if not found:
        return f"i could not find a place called {place}."
    spot = found[0]

    now = fetch_json("https://api.open-meteo.com/v1/forecast",
                     latitude=spot["latitude"], longitude=spot["longitude"],
                     current="temperature_2m,weather_code,wind_speed_10m",
                     temperature_unit="fahrenheit", wind_speed_unit="mph")["current"]
    fahrenheit = round(now["temperature_2m"])
    celsius = round((now["temperature_2m"] - 32) * 5 / 9)
    sky = SKY.get(now["weather_code"], "hard to describe")
    name = ", ".join(part for part in (spot.get("name"), spot.get("country")) if part)
    return (f"in {name} it is {fahrenheit}°F ({celsius}°C) and {sky}, "
            f"with wind at {round(now['wind_speed_10m'])} mph.")


# ----------------------------------------------------------------------------
# [lookup]
# ----------------------------------------------------------------------------
# Ways of asking that carry no information about the subject. Longest first.
LEAD_INS = sorted("""can you look up|can you tell me about|can you search for|what do you know about
|do you know about|do you know anything about|i want to know about|give me information about
|give me information on|tell me about|teach me about|find out about|facts about|info on|info about
|information about|information on|look up|search for|search|google|wikipedia|define|explain|describe
|what is|what's|whats|what are|what was|what were|who is|who's|whos|who was|who are|who were
""".replace("\n", "").split("|"), key=len, reverse=True)


def find_subject(message):
    """Turn a question into something to search for."""
    text = tidy(message)
    for lead_in in LEAD_INS:
        if text.startswith(lead_in + " "):
            return text[len(lead_in):].strip()
    return text


def lookup(message):
    subject = find_subject(message)
    if not subject:
        return "what should i look up? ask me like this: who was ada lovelace"

    # One request: search Wikipedia, and for the best matches return the
    # first two sentences of each article. Pages that only list the different
    # meanings of a word are skipped.
    answer = fetch_json("https://en.wikipedia.org/w/api.php",
                        action="query", format="json", formatversion=2,
                        generator="search", gsrsearch=subject, gsrlimit=3,
                        prop="extracts|pageprops", ppprop="disambiguation",
                        exintro=1, explaintext=1, exsentences=2)
    pages = sorted(answer.get("query", {}).get("pages", []), key=lambda page: page.get("index", 0))
    for page in pages:
        summary = " ".join(page.get("extract", "").split())
        is_list_of_meanings = "disambiguation" in page.get("pageprops", {})
        if summary and not is_list_of_meanings:
            return f"{summary} (from wikipedia: {page['title']})"
    return f"i could not find anything about {subject}."


# ----------------------------------------------------------------------------
# [news]
# ----------------------------------------------------------------------------
def news(message):
    feed = ET.fromstring(fetch("https://feeds.npr.org/1001/rss.xml"))
    titles = [item.findtext("title", "").strip() for item in feed.iter("item")]
    titles = [title for title in titles if title][:HEADLINES]
    if not titles:
        return "i could not find any headlines right now."
    lines = [f"{number}. {title}" for number, title in enumerate(titles, 1)]
    return "the top stories from npr right now:\n" + "\n".join(lines)


# ----------------------------------------------------------------------------
# Connecting the brain to the tools
# ----------------------------------------------------------------------------
TOOLS = {"[time]": tell_time, "[date]": tell_date, "[weather]": weather, "[lookup]": lookup, "[news]": news}


def use_tools(message, brain_reply):
    """Run a tool if the brain asked for one.

    Returns the text to show the person, and the name of the tool that was
    used (or None if the brain simply answered in words).
    """
    command = brain_reply.strip()
    if not command.startswith("["):
        return brain_reply, None

    # The brain writes one letter at a time, so now and then it misspells a
    # command. Accept the closest real one.
    close = difflib.get_close_matches(command, list(TOOLS), n=1, cutoff=0.6)
    if not close:
        return "sorry, i got confused. try asking another way.", None
    name = close[0].strip("[]")
    try:
        return TOOLS[close[0]](message), name
    except (urllib.error.URLError, TimeoutError):
        return f"i could not reach the internet for my {name} tool. are you online?", name
    except Exception as error:
        if str(error) == "certificates":
            return ("i cannot open secure websites yet. in terminal, run: "
                    "pip3 install certifi   and then try again."), name
        # Anything else is a bug worth seeing, so say exactly what went wrong.
        return f"my {name} tool hit a problem ({type(error).__name__}: {error}).", name

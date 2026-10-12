# tiny

A chatbot brain built from scratch, with tools for the time, the weather, the
news and Wikipedia.

## Run it

Open Terminal, then:

    cd ~/Desktop/TinyChat
    pip3 install numpy          (only needed once)

| To do this                        | Type this                    |
|-----------------------------------|------------------------------|
| Chat in Terminal                  | `python3 tinychat.py chat`   |
| Chat on a web page                | `python3 webchat.py`, then open http://localhost:8000 |
| Retrain the brain (about 20 min)  | `python3 tinychat.py train`  |
| Rebuild the training text         | `python3 make_data.py`       |

## What each file is

| File                   | What it is |
|------------------------|------------|
| `tinychat.py`          | The brain: the model, its training, and the chat loop |
| `tools.py`             | The tools: clock, weather, Wikipedia, news |
| `webchat.py`           | A small server: the chat page plus an API for your own apps |
| `chat_page.html`       | The chat page. Edit it to make it yours |
| `make_data.py`         | Writes the training text. Add topics here |
| `chat_data.txt`        | The training text |
| `tinychat_weights.npz` | The trained brain |
| `requirements.txt`     | Tells a hosting service to install NumPy |

## How the internet part works

The brain cannot go online. It has learned that some messages need a tool, and
for those it replies with a command instead of a sentence:

    you: what is the weather in tokyo
    brain: [weather]

`tools.py` sees `[weather]`, finds the place name in your message, asks a
weather service, and writes the sentence you see. The brain chooses the tool;
ordinary code does the fetching. The big chatbots reach the internet the same
way.

## Use it in your own app or website

Start `python3 webchat.py` and leave it running. Anything that can make a web
request can now talk to the bot:

    send      POST http://localhost:8000/chat    {"message": "tell me a joke"}
    receive   {"reply": "why was the math book sad? ...", "tool": null}

From a web page (JavaScript):

    const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "weather in tokyo" }),
    });
    const answer = await response.json();   // answer.reply, answer.tool

From an iPhone or Mac app (Swift). This snippet has not been run in Xcode yet:

    struct BotAnswer: Decodable { let reply: String; let tool: String? }

    func askBot(_ message: String) async throws -> BotAnswer {
        var request = URLRequest(url: URL(string: "http://localhost:8000/chat")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["message": message])
        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(BotAnswer.self, from: data)
    }

Apps block plain `http://` addresses unless you allow them. In Xcode, open the
target's Info tab and add "App Transport Security Settings" with "Allow Local
Networking" set to YES.

From another Python program, no server needed:

    from tinychat import Bot
    bot = Bot()
    print(bot.ask("who was ada lovelace"))

## Put it online for free (Render)

Render runs small programs like this one at no cost. You need a free GitHub
account and a free Render account.

1. On github.com, make a new repository and upload every file in this folder.
2. On render.com, choose New, then Web Service, and pick that repository.
3. Fill in these settings:

       Language         Python 3
       Build Command    pip install -r requirements.txt
       Start Command    python3 webchat.py
       Instance Type    Free

4. Press Deploy. After a few minutes Render shows your address, which looks
   like `https://your-name.onrender.com`. That address is your chat page, and
   `https://your-name.onrender.com/chat` is the API for your apps.

What free means on Render: the bot goes to sleep after 15 minutes without
visitors, and the next visitor waits about a minute while it wakes up.

Once it is online, use the `https://...onrender.com/chat` address in your app
in place of `http://localhost:8000/chat`. Because it starts with `https`, the
"Allow Local Networking" setting in Xcode is no longer needed.

## Limits to know about

- On your Mac the bot only works while `webchat.py` is running. `localhost`
  means "this computer", so the iPhone simulator can reach it but a real phone
  or a public website cannot. For a phone on your wifi, run
  `python3 webchat.py --network`. For everyone else, put it online as above.
- Once it is online, anyone with the address can talk to it.
- The time and date tools use the time zone named in `tools.py` (TIMEZONE).
- Lookups return the opening of the best matching Wikipedia article. The bot
  does not read or reason about it.
- Weather is the current conditions only, not a forecast.
- Each reply looks only at your latest message. It has no memory.

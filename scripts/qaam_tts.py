

from watson_text_talker import TextTalker, TT_Config, TT_Importance

from ..qaam import QAAM

TTSConfig = {
    "apikey": "KimsTEZMLu5TPUmS62utcjbt-ywpu95inG1cvqW43ROL",
    "iam_apikey_name": "maxqmodel",
    "url": "https://api.us-south.text-to-speech.watson.cloud.ibm.com/instances/7b0d5686-5cb2-4bad-bc78-796b95de05cc"
}

config = TT_Config()
config.API_KEY = TTSConfig["apikey"]
config.API_URL = TTSConfig["url"]
config.TTS_VOICE = "en-US_MichaelV3Voice"
config.INITIALIZATION_DELAY = 1
text_talker = TextTalker(config=config)

phrase = TT_Importance()
introduction = [
    (phrase.SAY_ALWAYS, "Hello, I am David."),
    (phrase.SAY_ALWAYS, "I am ready to answer your questions!"),
    (phrase.SAY_ALWAYS, "Please enter the website in the input field to get started.")
]

if introduction:
    text_talker.say_group(introduction)

blog_url = input("URL: ").strip()
if not blog_url.startswith("http"):
    print("Loading demo instead: Engine-Mount-Replacement")
    blog_url = "https://www.rmeuropean.com/bmw-e46-engine-mount-replacement.aspx"


qaam = QAAM(top_k=13)
qaam.add_url("http://25665f7a.ngrok.io")
qaam.texts_from_url(blog_url)

num_words = "Extracted {} words from the website!".format(len(qaam.doc))
if num_words:
    text_talker.say(num_words)


if __name__ == "__main__":
    STOP_FLAG = 'quit'
    while True:
        query = input("question : ").strip()
        if query.lower() != STOP_FLAG:
            q = query if query.endswith("?") else query.capitalize() + "?"
            prediction = qaam.answer(q)
            answer, context = prediction["answer"], prediction["context"]
            print("+ Answer: {}\n\nContext: {}\n".format(answer, context))
            if len(context.strip()) >= 255:
                text_talker.say(answer)
            else:
                text_talker.say(context)

        elif query.lower() == STOP_FLAG or KeyboardInterrupt:
            break
        else:
            continue

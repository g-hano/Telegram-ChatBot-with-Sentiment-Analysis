import io
import pandas as pd
from typing import Final
import openai
from telegram import Update,ChatPermissions
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackContext
import time
import re
import requests
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
import torch
import imageio
from newsdataapi import NewsDataApiClient
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import matplotlib.pyplot as plt

from tokens import (huggingFace_token, CHATGPT_APIKEY, NEWS_API_KEY,BOT_TOKEN,BOT_USERNAME,NASA_API)

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda"
token = huggingFace_token

pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=token)
pipe.to(device)

def image_Generator(phrase="a dog"):
    with autocast(device):
        image = pipe(phrase, guidance_scale=8.5).images[0]

    return image
    # pass
def video_generator(prompt="a dog eating pizza"):
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    video_frames = pipe(prompt, num_inference_steps=25).frames

    return video_frames
    # pass
def frames_to_video(frames, output_path='output_video.mp4', fps=25):
    imageio.mimwrite(output_path, frames, fps=fps, macro_block_size=None)
    # pass

ANALYSIS = []

news_api_key = NEWS_API_KEY
url = "https://api.newsapi.org/v2/top-headlines?apiKey=" + news_api_key

CHATGPT_APIKEY= CHATGPT_APIKEY

TOKEN: Final = BOT_TOKEN
BOT_USERNAME: Final = BOT_USERNAME

NASA_API = NASA_API
NASA_HTTP = 'https://api.nasa.gov/planetary/apod'

CURSE_WORDS = ["curse1", "curse2"]
chat_log=[]

# Commands
async def start_command(update: Update):
    await update.message.reply_text("Hello! Start command detected!")
async def help_command(update: Update):
    await update.message.reply_text("/start -> Starts the Bot\n/nasa -> Get today's high quality image from NASA\n/generateimg -> Generate image by text-to-image PreTrained Model\n/generatevid -> Generate video by text-to-video PreTrained Model\nAskGPT -> Use OpenAI's ChatGPT 3.5 model to answer\n/news -> search hottest news in the internet, example usage '/news ronaldo gb sports en'\nThe Model: 'https://huggingface.co/runwayml/stable-diffusion-v1-5'")
async def nasa_command(update: Update):

    parameters = {
        "api_key": NASA_API,
        "hd": True
    }
    # getting data from API
    response = requests.get(NASA_HTTP, params=parameters)
    response.raise_for_status()
    data = response.json()

    # creating png url
    image_url = data["hdurl"]

    # answer the comment
    await update.message.reply_text(f"Nasa command detected!\nNASA Image of the Day:  + {image_url}")
    # pass
async def generate_img_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Extract the command arguments from the message
    command_args = context.args

    # Join the command arguments to create the phrase
    phrase = " ".join(command_args)

    try:
        # Generate the image using the image_Generator function
        generated_image = image_Generator(phrase)

        # Convert the PIL Image to bytes
        image_byte_array = io.BytesIO()
        generated_image.save(image_byte_array, format='PNG')
        image_byte_array.seek(0)

        # Send the image to the Telegram chat
        await update.message.reply_photo(photo=image_byte_array)

        print("Image generated and sent!")

    except Exception as e:
        print(f"Error while generating and sending image: {e}")
        await update.message.reply_text("Error occurred while generating and sending the image.")
    # pass
async def generate_vid_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Extract the command arguments from the message
    command_args = context.args

    # Join the command arguments to create the phrase
    phrase = " ".join(command_args)

    try:
        # Generate the video using the video_generator function
        video_frames = video_generator(phrase)

        # Convert frames to video
        frames_to_video(video_frames, output_path='output_video.mp4')

        # Send the video to the Telegram chat
        with open('output_video.mp4', 'rb') as video_file:
            await update.message.reply_video(video=video_file)

        print("Video generated and sent!")

    except Exception as e:
        print(f"Error while generating and sending video: {e}")
        await update.message.reply_text("Error occurred while generating and sending the video.")
#     pass
async def chatgpt_command(update: Update, user_message=""):
    try:
        openai.api_key = CHATGPT_APIKEY
        response=openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}]
        )
        assistant_response = response["choices"][0]["message"]["content"]
        await update.message.reply_text(f"ChatGPT:", assistant_response.strip("\n").strip())
        chat_log.append({"role": "assistant", "content": assistant_response.strip("\n").strip()})
    except Exception as e:
        print(f"Error while starting ChatGPT: {e}")
        await update.message.reply_text(f"Error occurred while starting ChatGPT.\n{e}")
async def news_command(update: Update, user_message="ronaldo gb sports en"): ##
    try:
        parts = user_message.split(" ")
        key_word, country, category, language = parts

        api = NewsDataApiClient(apikey=news_api_key)

        response = api.news_api(q=key_word, country=country, category=category, language=language)
        # Extract the 'results' list from the response
        news_items = response['results'][:3]
        title = news_items[0].get('title')
        description = news_items[0].get('description')
        link = news_items[0].get('link')

        await update.message.reply_text(f"{title}\n{description}\n{link}")
    except Exception as e:
        await update.message.reply_text(f"Error!: {e}")
def preprocess_text(context: ContextTypes.DEFAULT_TYPE) -> str:
    text = context.args
    punctuation = string.punctuation
    return re.sub(r'[^\w\s{}]'.format(punctuation), '', str(text))

def fit_vectorizer():
    x_train = pd.read_csv('x_train.csv')
    vectorizer = TfidfVectorizer()
    vectorizer.fit(x_train)
    return vectorizer

def extract_words_inside_brackets(input_string):
    result = re.findall(r'\[(.*?)\]', input_string)
    if result:
        return result[0]
    return ""

async def analysis(update: Update, input_text:str):
    message_type: str = update.message.chat.type
    if message_type == "private":
        try:
            with open("model.pkl", 'rb') as f:
                model = pickle.load(f)
            with open("vectorizer.pkl", 'rb') as f:
                vectorizer = pickle.load(f)
        except FileNotFoundError:
            await update.message.reply_text("Error: Model not found.")
            return

        try:
            processed = preprocess_text(input_text)
            vector = vectorizer.transform([processed])

            predictions = model.predict(vector)
            print(f"Input Text: {input_text}\nPredicted Sentiment: {predictions}")

            global ANALYSIS
            sentiment = extract_words_inside_brackets(str(predictions))

            data = {
                'chat_id': update.effective_user.id,
                'sentiment': sentiment
            }

            ANALYSIS.append(data)
            print("\n\n\n\n\n", sentiment)
            await update.message.reply_text("Thank you for giving me information. I saved the prediction of your mood.")

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
    else:
        await update.message.reply_text(f"You can use that command only in private chats!")



async def mymood(update: Update):
    message_type: str = update.message.chat.type
    if message_type == "private":
        user_chat_id = update.effective_user.id
        user_analysis = [value for value in ANALYSIS if value['chat_id'] == user_chat_id]

        if not user_analysis:
            await update.message.reply_text("You don't have any saved mood analysis data.")
            return

        my_dict = {}
        for i, value in enumerate(user_analysis, 1):
            key = f"Day {i}"
            my_dict[key] = value['sentiment']

        # Mapping of sentiment strings to labels (adjust as per your sentiment categories)
        sentiment_labels = {
            "'joy'": "Joy",
            "'sadness'": "Sadness",
            "'anger'": "Anger",
            "'fear'": "Fear",
            "'neutral'": "Neutral"
        }

        # Count the occurrences of each sentiment category in the user's ANALYSIS
        sentiment_counts = {}
        for sentiment in user_analysis:
            sentiment_counts[sentiment['sentiment']] = sentiment_counts.get(sentiment['sentiment'], 0) + 1

        # Create the pie plot
        plt.figure(figsize=(8, 8))
        plt.pie(sentiment_counts.values(), labels=[sentiment_labels[s] for s in sentiment_counts], autopct='%1.1f%%')
        plt.title('My Mood Analysis')

        # Save the graph to a bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        # Send the graph image as a photo
        await update.message.reply_photo(photo=buffer)

    else:
        await update.message.reply_text("You can use that command only in private chats.")



# Responses
def handle_responses(text: str):
    processed: str = text.lower()

    if "SPECIALRESPONSE1" in processed:
        return "XXXX"

    if "SPECIALRESPONSE2" in processed:
        return "XXXXX"

    if "SPECIALRESPONSE3" in processed:
        return "XXXXXX"

    if "SPECIALRESPONSE4" in processed:
        return "XXXXXX"

    if processed in CURSE_WORDS:
        pass

    return "I understand nothing"
def contains_curse_word(text):
    # Function to check if the given text contains any curse word from the list
    for word in CURSE_WORDS:
        # Use regular expression for case-insensitive search
        if re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE):
            return True
    return False
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Get basic info of the incoming message
    message_type: str = update.message.chat.type
    text: str = update.message.text
    user_id: int = update.message.from_user.id

    # Print a log for debugging
    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')


    if contains_curse_word(text):
        try:
            chat_id = update.message.chat_id

            # Freeze the user from sending messages for a certain period
            until_date = int(time.time() + 30)
            context.bot.restrict_chat_member(
                chat_id=chat_id,
                user_id=user_id,
                permissions=ChatPermissions(can_send_messages=False),
                until_date=until_date
            )

            # Notify the user
            await update.message.reply_text(f"You have been banned for 30 sec")
        except Exception as e:
            print(f"Error while banning: {e}")


    # React to group messages only if users mention the bot directly
    if message_type == 'supergroup': # you can change it to private or group
        # Replace with your bot username
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_responses(new_text)
        else:
            return  # We don't want the bot respond if it's not mentioned in the group
    else:
        response: str = handle_responses(text)

    # Reply normal if the message is in private
    print('Bot:', response)
    await update.message.reply_text(response)
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")


if __name__ == "__main__":
    print("Starting bot!!!")
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("nasa", nasa_command))
    app.add_handler(CommandHandler("generateimg", generate_img_command))
    app.add_handler(CommandHandler("generatevid", generate_vid_command))
    app.add_handler(CommandHandler("askgpt", chatgpt_command))
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("diary", analysis))
    app.add_handler(CommandHandler("mymood", mymood))
    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors
    app.add_error_handler(error)

    # Polls the bot
    print("Polling...")
    app.run_polling(poll_interval=3)


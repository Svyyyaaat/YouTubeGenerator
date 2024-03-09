from openai import OpenAI
from anthropic import Anthropic
import os
import asyncio
import edge_tts
import ffmpeg
from PIL import Image
from multiprocessing import Pool
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import AudioFileClip
from moviepy.editor import concatenate_videoclips
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.fx.audio_fadein import audio_fadein
from moviepy.audio.fx.audio_fadeout import audio_fadeout
from moviepy.audio.fx.volumex import volumex
from random import sample, randint, choice
import stable_whisper
import gradio as gr

# from TTS.utils.manage import ModelManager
# from TTS.utils.synthesizer import Synthesizer

OPENAI_TOKEN = ''  # insert openai token here
CLAUDE_TOKEN = ''
FONT_PATH = r'fonts\unicode.futurab.ttf'  # insert path to font file (for thumbnail gen)
MODEL = "claude" # claude or chatgpt (copy and paste)

def ask(question):
    if MODEL == 'claude':
        client = Anthropic(
            # This is the default and can be omitted
            api_key=CLAUDE_TOKEN,
        )
        message = client.messages.create(
            #model="claude-instant-1.2",
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return list(message.content[0])[0][1]
    elif MODEL == 'chatgpt':
        client = OpenAI(api_key=OPENAI_TOKEN)
        answer = client.chat.completions.create(model='gpt-4', messages=[
            {"role": "user", "content": question}]).choices[0].message.content
        return (answer)



# make a speech
def get_speech(channel):
    # in case of network errors, functions may try again
    for i in range(0, 1):
        try:
            channel_name = channel[:channel.find(' - ')]
            channel_theme = channel[channel.find(' - ') + 3:-4]
            with open(f'channels/{channel}') as f:
                used_topics = f.readlines()

            # getting a topic
            t = 0
            while t <= 3:
                t+=1
                print(f"making a topic for the channel {channel_name}")
                if len(used_topics) != 0:
                    used_topics_str = '"' + '", "'.join(used_topics) + '"'
                    topic_request = f'Come up with 5 interesting and original viral video ideas for a YouTube channel dedicated to the theme of "{channel_theme}". Think who can be the target audience of the channel and make ideas according to it. Don\'t use the following topics: {used_topics_str}'
                else:
                    topic_request = f'Come up with 5 interesting and original viral video ideas for a YouTube channel dedicated to the theme of "{channel_theme}". Think who can be the target audience of the channel and make ideas according to it. In your answer write only the list of topics, don\'t describe them.'
                suggested_topics = ask(topic_request)
                main_topic = ask(
                    f'Choose the best topic for a viral YouTube video of a channel dedicated to {channel_theme} among the ones listed here: "{suggested_topics}". In your answer write only the topic. Refrase it so there will be ":", like in "Relationships Empathy: Nurturing Healthy Connections". Make the topic concise.')

                if len(main_topic) < 100:
                    print(f"{main_topic} is the topic for the channel {channel_name}")
                    t = 1000
                    break
                else:
                    print(f"making another topic for the channel {channel_name}")
            if t != 1000:
                raise IndexError
            # writing a speech
            t = 1
            match = ['canada']
            while t <= 5:
                speech_request = f'Write an extremely short speech (no longer than 200 words) for a 30 seconds YouTube video on the topic of {main_topic}. The channel name is {channel_name} and it\'s dedicated to {channel_theme}. Make a unique practical advice accompanied by a story explaining why it is important to follow the advice and how to follow it. The story must be reasonable and believable. The story must be concise, practical, and interesting in itself, just like a novel or a tale. Don\'t write stuff like "Sarah and Mark had a disagreement about their future plans, but by active listening they found a compromise", write stuff like "Sarah wanted to move to Canada from the US, and Mark wanted to stay at the US. Instead of arguing why it\'s better to stay at the US, Mark tried to understand why Sarah tried to leave in the first place. It turned out that she wanted to get as far as possible from her toxic family, so Mark helped to make the relationship with her family better, therefore removing the need to move from the US". Do not use this exact story or one with a similar plot. It is very important that you don\'t use this exact story or one with a similar plot. Use names in your stories. Use specifics. Keep the text concise. End the text when the story ends, don\'t explain it too much. Don\'t say "embark on a journey", "world", "wonderful", "captivating", "thrilled", "fascinating", and similar strong adjectives, use something more relaxed. The host is a woman with MBTI INFJ type, genuinely caring about her viewers and willing to improve their life. Make the speech sound conversational. Call viewers simply "viewers of {channel_name}", without any adjectives. Don\'t write a script, write only the speech. I want you to simply write the text for reading without mentioning actions or roles. Don\'t put in stuff like "music playing" or "host: %host speech%". Give me the title, the speech, and your comments if you have any in the following format: "Title: ... \nSpeech: ... \n{MODEL} comment: ...'

                speech = ask(speech_request)

                #update_speech_request = f'I have a script Claude wrote for a YouTube video. ' \
                #                        f'Replace strong words like "embark on a journey", "world", "wonderful", "captivating", "thrilled", ' \
                #                        f'"fascinating" with something more natural and conversation-like. ' \
                #                        f'Call viewers simply "viewers of {channel_name}", without any adjectives, if you see stuff like ' \
                #                        f'"lovely viewers" - replace it. ' \
                #                        f'If there\'s any story about Sarah, Mark, or about someone moving from one country to another, ' \
                #                        f'replace it with a completely different story. That\'s the most important part of your task.' \
                #                        f'\nKeep the text extremely short, no longer than 30 seconds to read' \
                #                        f'\nGive me the title, the speech, ' \
                #                        f'and your comments if you have any in the following format: ' \
                #                        f'"Title: ... \nSpeech: ... \nClaude comment: ...\nHere\'s the original script: "{speech}"'

                # update_speech = ask(update_speech_request)
                update_speech = speech
                speech_start = update_speech.lower().find('speech:')
                speech_fin = update_speech.lower().find(f'{MODEL} comment:')
                if speech_fin == -1:
                    update_speech = update_speech[speech_start + 7:]
                else:
                    update_speech = update_speech[speech_start + 7: speech_fin]
                if any(c in update_speech.lower() for c in match):
                    t += 1
                    print(f'writing another speech for {channel_name}')
                else:
                    t = 1000
                    print(f'speech on the topic {main_topic} for the channel {channel_name} complete')
                    # adding the topic to "used topics"
                    with open(f'channels/{channel}', 'a') as f:
                        f.write(main_topic.replace('"', '') + '\n')
            if t != 1000:
                print(f'too many attempts to write a speech for {channel_name}')
                raise IndexError

            ret = [channel, main_topic, update_speech]
            return (ret)

        except Exception as fuu:
            print(f'error {fuu}, restarting for {channel_name}')
        else:
            break


# make an audio (woman voice)
# silenced function below was made before I discovered edge-tts
# it's slow and the voice is robotic
"""
def dep_say_w(channelXfile_nameXspeech):
    text = channelXfile_nameXspeech[2]
    file_name = channelXfile_nameXspeech[1]
    channel = channelXfile_nameXspeech[0][:-4]
    print(f"starting audio on topic {file_name} for channel {channel}")
    file_name = file_name.replace('"', '')
    file_name = file_name.replace(':', '')
    file_name = file_name.replace('|', '')
    file_name = file_name.replace('/', '')
    file_name = file_name.replace('\\', '')
    file_name = file_name.replace('?', '')
    file_name = file_name.replace('<', '')
    file_name = file_name.replace('>', '')
    text.replace(' - ', ', ')
    text.replace(' – ', ', ')
    text.replace(' — ', ', ')
    text.replace(' - ', ', ')
    text.replace(':', '.')
    file_full_name = f"audio/{channel}/" + file_name + '.wav'
    path = "C:/Users/USER_NAME/AppData/Local/Programs/Python/Python310/Lib/site-packages/TTS/.models.json" # insert username

    model_manager = ModelManager(path)

    model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC_ph")

    voc_path, voc_config_path, _ = model_manager.download_model(model_item["default_vocoder"])

    syn = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        vocoder_checkpoint=voc_path,
        vocoder_config=voc_config_path
    )

    outputs = syn.tts(text)
    syn.save_wav(outputs, file_full_name)

    print(f"audio on topic {file_name} for channel {channel} complete")
    """


def say_w(channelXfile_nameXspeech):
    """
    generates voice for a list containing:
    [0] channel name (with .txt on the end)
    [1] topic of a video
    [2] speech
    """
    text = channelXfile_nameXspeech[2]
    file_name = channelXfile_nameXspeech[1]
    channel = channelXfile_nameXspeech[0][:-4]
    voice = "en-GB-SoniaNeural"
    print(f"starting audio on topic {file_name} for channel {channel}")
    file_name = file_name.replace('"', '')
    file_name = file_name.replace(':', ';')
    file_name = file_name.replace('|', '')
    file_name = file_name.replace('/', '')
    file_name = file_name.replace('\\', '')
    file_name = file_name.replace('?', '')
    file_name = file_name.replace('<', '')
    file_name = file_name.replace('>', '')
    file_full_name = f"audio/{channel}/" + file_name + '.mp3'

    async def amain() -> None:
        communicate = edge_tts.Communicate(text=text, voice=voice, volume="+40%", rate="+8%")
        await communicate.save(file_full_name)

    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.run_until_complete(amain())
    finally:
        loop.close()
    print(f"made audio on topic {file_name} for channel {channel}")


def get_thumbnail(text):
    animals_shock_raw = os.scandir('thumbnail/animals_shock')
    backgrounds_raw = os.scandir('thumbnail/backgrounds')

    backgrounds_names = []
    for entry in backgrounds_raw:
        backgrounds_names.append(entry.path)

    animals_shock_names = []
    for entry in animals_shock_raw:
        animals_shock_names.append(entry.path)

    names = animals_shock_names

    # breaking the text
    end_pt = text.find(';')
    if end_pt != -1:
        text = text[:end_pt]

    text_list = text.split()

    # writing the text
    i = 0
    while i < len(text_list):
        if i + 1 < len(text_list):
            if len(text_list[i] + ' ' + text_list[i + 1]) <= 28:
                text_list[i] = text_list[i] + ' ' + text_list[i + 1]
                text_list.remove(text_list[i + 1])
            else:
                i += 1
        else:
            i += 1

    biggest_line = max(text_list, key=len)
    for i in range(len(text_list)):
        line = text_list[i]
        if len(line) < len(biggest_line):
            add = int((len(biggest_line) - len(line)) / 2)
            line = add * ' ' + line
            text_list[i] = line

    string = '\n'.join(text_list)
    print(string)

    if len(string) >= 22:
        FONT_SIZE = 80
    elif len(string) >= 18:
        FONT_SIZE = 100
    else:
        FONT_SIZE = 120

    ffmpeg.input(choice(backgrounds_names)).filter(
        'drawtext',
        text=string,
        x='(w-text_w)/2',
        y=80,
        fontcolor='d1e8d7',
        boxcolor='black',
        box=1,
        boxborderw='20',
        fontsize=FONT_SIZE,
        fontfile=FONT_PATH
    ).output(r'thumbnail\text_image.png').run()

    # combining background and rabbit
    background = Image.open(r'thumbnail\text_image.png')
    overlay = Image.open(choice(names))
    height = overlay.height
    width = overlay.width
    new_width = 720
    new_height = int(new_width * height / width)
    overlay = overlay.resize((new_width, new_height), Image.LANCZOS)
    background.paste(overlay, (300, 180), overlay)
    background.save(f'thumbnail\\{text}.png')

    os.remove(r'thumbnail\text_image.png')


# make a video (having the audio)
def make_video(channelXfile_nameXspeech):
    file_name = channelXfile_nameXspeech[1]
    channel = channelXfile_nameXspeech[0][:-4]
    file_name = file_name.replace('"', '')
    file_name = file_name.replace(':', ';')
    file_name = file_name.replace('|', '')
    file_name = file_name.replace('/', '')
    file_name = file_name.replace('\\', '')
    file_name = file_name.replace('?', '')
    file_name = file_name.replace('<', '')
    file_name = file_name.replace('>', '')

    speech_audio = AudioFileClip(f"audio/{channel}/{file_name}.mp3")
    speech_duration = speech_audio.duration
    clips_amount = int(speech_duration // 3)

    nature_raw = os.scandir('nature_primitive')
    nature_names = []
    for entry in nature_raw:
        nature_names.append(entry.path)

    couples_raw = os.scandir('couples_primitive')
    couples_names = []
    for entry in couples_raw:
        couples_names.append(entry.path)

    people_chill_raw = os.scandir('people_chill_primitive')
    people_chill_names = []
    for entry in people_chill_raw:
        people_chill_names.append(entry.path)

    people_work_raw = os.scandir('people_work_primitive')
    people_work_names = []
    for entry in people_work_raw:
        people_work_names.append(entry.path)

    cities_raw = os.scandir('cities_primitive')
    cities_names = []
    for entry in cities_raw:
        cities_names.append(entry.path)

    if channel in ['Love Code - Psychology of Happy Relationships']:
        clip_names = nature_names + couples_names + people_chill_names
    elif channel in ['Empathy Elevator - Emotional Intelligence Development']:
        clip_names = nature_names + cities_names + people_chill_names + people_work_names
    elif channel in ['Empathy for Kids - Teaching Empathy and Emotional Intelligence to young children']:
        clip_names = nature_names + people_chill_names
    else:
        clip_names = nature_names + people_chill_names + people_work_names + cities_names
    clips_for_use = sample(clip_names, clips_amount)

    # now let's add music, speech and subs
    # making a silent & non-subbed video

    clips = []
    for c in clips_for_use:
        c = VideoFileClip(c)
        clips.append(c)
    final_video = concatenate_videoclips(clips)

    final_video.write_videofile(f'video/{channel}/{file_name}_silent_overlayed.mp4', threads=32, fps=30)

    # breaking the text
    text = file_name
    end_pt = text.find(';')
    if end_pt != -1:
        text = text[:end_pt]

    text_list = text.split()

    # putting intro overlay
    # (didn't turn out well so turned it off)
    """
    animals_shock_raw = os.scandir('thumbnail/animals_shock')
    backgrounds_raw = os.scandir('thumbnail/backgrounds')

    backgrounds_names = []
    for entry in backgrounds_raw:
        backgrounds_names.append(entry.path)

    animals_shock_names = []
    for entry in animals_shock_raw:
        animals_shock_names.append(entry.path)

    animals_names = animals_shock_names

    (
        ffmpeg
        .filter([ffmpeg.input(f'video/{channel}/{file_name}_silent.mp4'), ffmpeg.input(r'thumbnail\\frame.png')],
                'overlay', 9999, 9999,
                enable='between(t, 0, 6)')
        .output(f'video/{channel}/{file_name}_silent_frame.mp4')
        .run()
    )
    # writing the text
    i = 0
    while i < len(text_list):
        if i + 1 < len(text_list):
            if len(text_list[i] + ' ' + text_list[i + 1]) <= 28:
                text_list[i] = text_list[i] + ' ' + text_list[i + 1]
                text_list.remove(text_list[i + 1])
            else:
                i += 1
        else:
            i += 1

    string = '\n'.join(text_list)
    print(string)

    if len(string) >= 22:
        FONT_SIZE = 80
    elif len(string) >= 18:
        FONT_SIZE = 100
    else:
        FONT_SIZE = 120

    lines_count = len(text_list)
    for i in range(lines_count):
        if i == 0 and i != lines_count - 1:
            base_video = f'video/{channel}/{file_name}_silent_frame.mp4'
            output_video = f'video\\{channel}\\{file_name}_silent_frame_text{i}.mp4'
        elif i == 0 and i == lines_count - 1:
            base_video = f'video/{channel}/{file_name}_silent_frame.mp4'
            output_video = f'video\\{channel}\\{file_name}_silent_frame_text.mp4'
        elif i != lines_count - 1:
            base_video = f'video\\{channel}\\{file_name}_silent_frame_text{i-1}.mp4'
            output_video = f'video\\{channel}\\{file_name}_silent_frame_text{i}.mp4'
        else:
            base_video = f'video\\{channel}\\{file_name}_silent_frame_text{i-1}.mp4'
            output_video = f'video\\{channel}\\{file_name}_silent_frame_text.mp4'
        ffmpeg.input(base_video).filter(
            'drawtext',
            text=text_list[i],
            #x='(w-text_w)/2',
            #y=80 + i * 100,
            x = 9999,
            y = 9999,
            fontcolor='d1e8d7',
            boxcolor='black',
            box=1,
            boxborderw='20',
            fontsize=FONT_SIZE,
            fontfile=FONT_PATH,
            enable='between(t, 0, 6)'
        ).output(output_video).run()

    # putting an animal
    overlay = Image.open(choice(animals_names))
    height = overlay.height
    width = overlay.width
    if len(text_list) <= 1:
        new_width = 720
        ycords = 160
    else:
        new_width = 640
        ycords = 200
    new_height = int(new_width * height / width)
    xcords = int((1280-new_width)/2)
    overlay = overlay.resize((new_width, new_height), Image.LANCZOS)
    overlay.save(r'video\\temp_rabbit.png')

    (
        ffmpeg
        .filter([ffmpeg.input(f'video\\{channel}\\{file_name}_silent_frame_text.mp4'),
                 ffmpeg.input(r'video\\temp_rabbit.png')],
                'overlay', 9999, 9999,
                enable='between(t, 0, 3)')
        .output(f'video/{channel}/{file_name}_silent_overlayed.mp4')
        .run()
    )
    """

    # making subs
    print(f"making subs for {file_name}")
    model = stable_whisper.load_model("medium", dq=True)
    result = model.transcribe(
        f"audio/{channel}/{file_name}.mp3",
        language='English')
    result.to_srt_vtt(f"audio/{channel}/{file_name}.srt")

    # getting music sample
    start_time = randint(0, 10000)
    music = AudioFileClip("lofi.mp3").subclip(start_time, start_time + clips_amount * 3)
    music = audio_fadein(music, 1)
    music = audio_fadeout(music, 1)
    music = volumex(music, 0.15)

    # writing complete audiofile
    audio = CompositeAudioClip([music, speech_audio])
    audio.write_audiofile(f"audio/{channel}/{file_name}_with_music.mp3")

    # combining silent&not-subbed video, audio and subs
    video = ffmpeg.input(f'video\\{channel}\\{file_name}_silent_overlayed.mp4')
    audio = ffmpeg.input(f"audio/{channel}/{file_name}_with_music.mp3")

    (
        ffmpeg
        .concat(
            video
            .filter('subtitles', filename=f'audio/{channel}/{file_name}.srt',
                    #        fontsdir="C:\Windows\Fonts",
                    force_style='Fontname=Futura,Fontsize=24,Outline=1,Shadow=2,BackColour=&H7C0B43&,OutlineColour=&H7C0B43&',
                    original_size='1280x720'),
            audio,
            v=1,
            a=1
        )
        .output(f'video/{channel}/{file_name}.mp4')
        .run()
    )

    # getting rid of composing files

    # os.remove(f'video/{channel}/{file_name}_silent.mp4')
    # os.remove(f'video/{channel}/{file_name}_silent_frame.mp4')
    # os.remove(f'video/{channel}/{file_name}_silent_frame_text.mp4')
    # for i in range(len(text_list)-1):
    #    os.remove(f'video/{channel}/{file_name}_silent_frame_text{i}.mp4')
    os.remove(f'C:video\\{channel}\\{file_name}_silent_overlayed.mp4')
    os.remove(f"audio/{channel}/{file_name}_with_music.mp3")
    os.remove(f"audio/{channel}/{file_name}.srt")
    os.remove(f"audio/{channel}/{file_name}.mp3")

    return(f'video/{channel}/{file_name}.mp4')


def check_and_create_file(directory, filename):
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
        except Exception as e:
            print(f"Error creating directory '{directory}': {e}")

    # Check if the file exists in the directory
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        return 1
    else:
        # If the file doesn't exist, create it
        try:
            with open(filepath, 'w'):
                pass
            print(f"File '{filename}' created successfully in '{directory}'.")
        except Exception as e:
            print(f"Error creating file '{filename}' in '{directory}': {e}")

def prepare_video(new_channel, choose_channel, amount=1):
    if new_channel != '':
        new_channel += '.txt'
        channel_list = [new_channel]
        # creating folders for first-time used channels
        if not check_and_create_file(directory='channels', filename=new_channel):
            check_and_create_file(directory=f'video/{new_channel[:-4]}', filename=new_channel)
            check_and_create_file(directory=f'audio/{new_channel[:-4]}', filename=new_channel)
    elif choose_channel != '':
        choose_channel += '.txt'
        channel_list = [choose_channel]
    else:
        print("Error: no input")
        return 0

    channel_list *= amount

    # writing text
    speech_pool = Pool()
    channelXtopicsXspeeches_list = speech_pool.map(get_speech, channel_list)
    speech_pool.close()
    speech_pool.join()

    # voicing text
    voice_pool = Pool()
    voice_pool.map(say_w, channelXtopicsXspeeches_list)
    voice_pool.close()
    voice_pool.join()

    # preparing videos
    video_paths = []
    for channelXtopicXspeech in channelXtopicsXspeeches_list:
        try:
            video_paths.append(make_video(channelXtopicXspeech))
        except Exception as ex:
            print(ex)
    while len(video_paths) < 5:
        video_paths.append("video/placeholder.mp4")
    return(video_paths)


def show_channels():
    channels = []
    for ch in os.listdir('channels'):
        channels.append(ch[:-4])
    return(channels)

if __name__ == "__main__":

    demo = gr.Interface(
        fn=prepare_video,
        inputs=[
            gr.Text(label="add new channel"),
            gr.Dropdown(choices=show_channels()),
            gr.Slider(1, 5, value=1, step=1, label="Amount", info="Amount of videos to generate")
        ],
        outputs=['video' for i in range(5)]
    )

    demo.launch()
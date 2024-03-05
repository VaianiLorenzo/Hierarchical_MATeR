import openai
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='ChatGPT')
parser.add_argument('--temp', type=float, default=0.0)
k_shot_parser = parser.add_mutually_exclusive_group(required=False)
k_shot_parser.add_argument('--k_shot', dest='k_shot', action='store_true')
parser.set_defaults(k_shot=False)
args = parser.parse_args()


deployment_id="gpt-35-turbo"
model="gpt-35-turbo"
temperature = args.temp
k_shot = args.k_shot

credentials_path = "credentials.json"
with open(credentials_path, "r") as f:
    credentials = json.load(f)
API_KEY = credentials['OPENAI_KEY']
API_BASE = credentials['OPENAI_BASE']

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = OPENAI_BASE
openai.api_version = "2023-05-15"

input_path = "output/Hierarchical-MATeR_test.json"

with open(input_path, 'r') as f:
    data = json.load(f)

k_shot_gts = [
    "In this episode, we interview our guest, Michele, who shares her story of overcoming childhood physical and sexual abuse. In addition, she has overcome breast cancer. She gives a beautiful and vulnerable testimony that gives hope to others overcoming similar issues.",
    "This podcast will share part of my story of why I left my classroom and how I ended up in my new job. This episode shares a lot of personal parts of my journey and I hope you don't feel so alone after listing. I have a free printable that you can hang in your classroom or home to remind you how strong you are.",
    "Your first impression is the most important impression.  That's why we have the founder of First Impression Jordan on our show to tell us how to make the best first impression IRL and online. He gives us the recipe for ultimate success when making a first impression.",
    "Today we discuss letting go on the things we cant control, and leaning into the things we can! Enjoy!", 
    "Complete fire on this episodes of the #podSessions. I had my man Gabe Anderson of VaynerBeta, Ultra athlete and author Rich Roll and Instagram hit / pro actress Arabella S Ruby on the show this week to chop it up about business, culture, social media, and a TON more. Great, great stuff on this show."
]

k_shot_senetnces = [
    "Talking about it and what she's doing or how she's gone forward focusing on a journey and how it we get to the other side and how we find our peacefully Ever After so welcome, Michelle.\nIt is it's given me freedom to give hope.",
    "Welcome back to another episode of the podcast. I don't think I've ever been more nervous to share something than I am right now. But what I have to share is such an important part of today's broadcast. I'm finally sharing parts of why I left my teaching job. And this was a doozy to reflect on I started teaching a classroom for students with autism in 2013. I made mistakes that many first-year teachers make and I struggled I slowly grew a little bit each year and I was doing\nwork with students with disabilities. I would have flexibility to work on my teachers pay teacher's store. So I took the leap and applied and I'm happy. I'm definitely happy. It's still hard. This was a really hard podcast to share because I was sharing a lot of things I never shared before but I wanted to share it for two main reasons first. I wanted you to know that if you are ever placed on probation or you're ever in a situation where",
    "How do you make a good one? How do you screw it up? We're going to dive into First Impressions those first 5 seconds 1 minute 5 minutes and figure out the recipe for success with First Impressions. So keep listening. Hey guys, welcome to the ask women podcast. I'm your host Kristen Carney along with \nOther host Marnie cameras, and today we have a dude. I'm very professional as we know. I have a dude who has created a new app for dating online. And so he's going to talk to us all about it and his name is Jordan, but I'm not going to say his last name because you'll think he's related to someone famous. Hey Jordan, thanks for joining us. Hey, how you guys doing today? We're good. Well, so we all know that the world needs another dating app, right, right. Yes, so",
    " That can totally relate with the lifestyle that we have which is constant busyness. So yeah, I'm today. We are going to talk how to accept not being in control which is a huge part of Farm Life. The first thing is what can you control and that is absolutely your attitude your mindset before when Bart and I first started farming\n Oh and welcome back to episode 3 of the Midwest Farm wives podcast Kylie and I have had a busy week. We had our County Fair and our kiddos did the Peewee Peewee showing and so they showed some pigs of my niece and nephews and then we had our bucket calf and my daughter did the rodeo the little britches Rodeo. So we were we were in town three days every single day morning and night too.",
    " weeks and top 100 virality people are watching it matters. This could be your moment. A lot of fun things covered on the show. Hope you enjoyed it Seth good job. We'll see you next time. I'm pod Sessions. Hey guys. I hope you really enjoyed this episode of The garyvee Experience now go out and share this pass it on. Let me know what you thought.\n I hope you enjoyed it voice cont'd I'm serious. I'm throwing a right hook at the end. I can just feel it in my bones that I'm about to Embark into the next Frontier of my career. It's around voice. It's podcasts Alexis skills. It's all that whether you come to the conference or not. Just dig in dig in. It's the opportunity to merging. It's really all of you are going to be using voiceover texting for a ton of things that you can be thinking about right now. So I'm just excited about it's that next order to having Rich here. I think is inspire me 2012 podcast three"
]


new_dict = {}
for key in tqdm(data.keys()):

    sentences = data[key]['2']    
    messages = []

    if k_shot:
        for sentence, gt in zip(k_shot_senetnces, k_shot_gts):
            prompt = "Create a description for a podcast that contains the following sentences:\n"
            prompt += sentence + "\n"
            prompt += "Description:"
            messages.append({"role": "user", "content": sentence})
            messages.append({"role": "assistant", "content": gt})
  
    prompt = "Create a description for a podcast that contains the following sentences:\n"    
    for sentence in sentences:
        prompt += sentence + "\n"
    prompt += "\nDescription:"
    messages.append({"role": "user", "content": prompt})
    try:
        response = openai.ChatCompletion.create(
                deployment_id=deployment_id,
                model=model,
                messages=messages,
                temperature=temperature
            )
        response = response.choices[0].message["content"]
        #print("Description:", response)
    except Exception as e:
        print("KEY: ", key)
        response = ""

    tmp_dict = {}
    tmp_dict['1'] = [response]
    new_dict[key] = tmp_dict

with open("output/abstractive-chatGPT_def_2_"+str(temperature)+"_5-shot_Hierarchical-MATeR_test.json", 'w') as f:
    json.dump(new_dict, f)




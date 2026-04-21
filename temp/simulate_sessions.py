"""
simulate_sessions.py

Runs 20 fully automated therapy sessions with diverse patient contexts.
Each session uses session_id=1 and saves logs to logs/{patient_id}_1.json.
Patient messages are pre-scripted to realistically progress through phases.

Run: python simulate_sessions.py
"""

import time
from pipeline.therapy_pipeline import therapy_chat, reset_for_new_patient

SESSION_ID = 1

# ------------------------------------------------------------------
# 20 diverse patient scenarios
# Each entry: patient_id, short description, list of patient turns
# Turns are designed to naturally progress: status → triggers → motivation → planning → closing
# ------------------------------------------------------------------

PATIENT_SCENARIOS = [
    {
        "patient_id": "sim_p01",
        "description": "Young male college student, social smoker, exam stress",
        "turns": [
            "Hi, I've been smoking cigarettes for about 2 years now, mostly socially.",
            "Mostly when I'm stressed about exams or hanging out with friends who smoke.",
            "I want to reduce first, not quit cold turkey. My parents don't know I smoke.",
            "I tried just avoiding parties for a week but it didn't help, I ended up smoking more during exams.",
            "I think I'm most ready to cut down on exam days. Maybe chewing gum could help?",
            "Yeah I think I can try replacing cigarettes with gum during study sessions. That sounds doable.",
            "Okay I'll give it a shot. Thank you.",
        ],
    },
    {
        "patient_id": "sim_p02",
        "description": "Middle-aged female, 20-year smoker, lung health concern",
        "turns": [
            "I've been smoking for almost 20 years, about a pack a day.",
            "My doctor told me last month my lung function is declining. That really scared me.",
            "I smoke the most after meals and when I'm bored at home in the evenings.",
            "I've tried nicotine patches twice. They helped a little but I always went back.",
            "I want to quit fully. My health is really my main reason, I have kids to take care of.",
            "Maybe the patches combined with something for the evening boredom? Like going for a walk?",
            "I think that's a solid plan. I'll try to replace my evening smoke with a 10-minute walk.",
            "Yes, I'm ready to start. I'll begin tomorrow morning.",
        ],
    },
    {
        "patient_id": "sim_p03",
        "description": "Elderly male, bidi smoker, family pressure to quit",
        "turns": [
            "I smoke bidis, have been for 35 years. About 15 to 20 bidis a day.",
            "My son keeps telling me to stop. My grandchildren are scared of getting sick around me.",
            "I smoke most after tea in the morning and after meals. It's become a habit I don't think about.",
            "I tried stopping once maybe 10 years ago but I couldn't sleep and felt very irritable.",
            "I want to stop for my family. I don't want my grandchildren to see me like this.",
            "Maybe reducing slowly would work better for me. Start with skipping the morning one.",
            "Yes, I can try skipping the bidi after morning tea for a week. That seems possible.",
        ],
    },
    {
        "patient_id": "sim_p04",
        "description": "Young female, 1-year smoker, wants to quit for pregnancy planning",
        "turns": [
            "I've only been smoking for about a year, maybe 4 to 5 cigarettes a day.",
            "My husband and I are planning to start a family soon and I know smoking is harmful.",
            "I usually smoke during breaks at work, it's a stress outlet honestly.",
            "I haven't really tried to quit before. This would be my first real attempt.",
            "I really want to quit completely before we start trying for a baby. That's my deadline.",
            "I could replace the work break cigarette with a short walk or breathing exercise.",
            "That sounds really practical. I'll start the breathing exercises tomorrow at my break.",
            "Thank you so much. I feel motivated now.",
        ],
    },
    {
        "patient_id": "sim_p05",
        "description": "Middle-aged male, heavy smoker, work stress and deadlines",
        "turns": [
            "I smoke about 25 cigarettes a day. Been doing this for 15 years.",
            "Work is insane. I'm in project management and the deadlines never stop.",
            "Every time there's a crisis at work, I just step outside and smoke. It's my only break.",
            "I tried cold turkey once. Made it 3 days and then had a terrible meeting and lit up again.",
            "I want to cut down for now. Even going from 25 to 10 would feel like a win.",
            "Maybe I can set a limit, like only smoking at designated times, not whenever stress hits.",
            "I'll try the scheduled smoking approach. Fix 5 times a day and not smoke outside that.",
            "Okay. I'll track it this week and report back. Thank you.",
        ],
    },
    {
        "patient_id": "sim_p06",
        "description": "Teenage male, peer pressure smoker, wants to quit for sports performance",
        "turns": [
            "I'm 17 and I've been smoking for about 8 months. My friends smoke so I started.",
            "I'm on the football team and my coach noticed I'm getting tired faster. He suspects something.",
            "I mostly smoke after school when we hang out near the ground. Feels like I fit in.",
            "I haven't tried quitting. I'm kind of scared to tell my friends I want to stop.",
            "I want to play better and not get dropped from the team. Football matters more to me.",
            "Maybe I can just say I'm following a fitness regime and can't smoke. That's even true.",
            "Yeah that excuse is honest and doesn't make me look weird. I'll use that this week.",
            "Alright. I'll try to avoid the smoke break and see how it goes. Thanks.",
        ],
    },
    {
        "patient_id": "sim_p07",
        "description": "Elderly female, hookah smoker, religious motivation",
        "turns": [
            "I smoke hookah at home every evening. I've been doing this for about 12 years.",
            "My religious teacher told me it's not permitted and I should stop. That stayed with me.",
            "I smoke mostly in the evening to relax. My husband passed away and it's a lonely time.",
            "I tried once during Ramadan and managed for the month but went back after.",
            "I want to stop because of my religion and also I've been having chest problems recently.",
            "Perhaps I can replace the evening hookah with some prayer time or reading. That would help.",
            "Yes, that is a good idea. I will try to spend that time reading instead.",
            "Thank you. I will make this effort.",
        ],
    },
    {
        "patient_id": "sim_p08",
        "description": "Young male, chain smoker, night owl with boredom triggers",
        "turns": [
            "I smoke probably 30 cigarettes a day. I stay up very late and smoke when I'm bored at night.",
            "I work from home as a freelance developer. I code late into the night and just keep smoking.",
            "Boredom and having nothing else to do with my hands while thinking through code problems.",
            "I tried nicotine gum but I found it disgusting. Patches made my skin itch.",
            "Honestly I just want to feel healthier. I wake up coughing every day. It's getting bad.",
            "Maybe I can try fidget tools or a stress ball to keep my hands busy while coding.",
            "That actually makes sense. I'll order a fidget cube tonight and see if it helps at all.",
            "Okay I'll try that first and see. Thanks.",
        ],
    },
    {
        "patient_id": "sim_p09",
        "description": "Middle-aged female, occasional smoker, triggers around alcohol",
        "turns": [
            "I'm more of a social smoker, maybe 3 to 4 cigarettes a week, mostly on weekends.",
            "Whenever I drink at a party or dinner, I almost always end up smoking with others.",
            "The combination of alcohol and social pressure makes it feel natural to smoke.",
            "I've never really tried to quit formally because it doesn't feel like a full addiction.",
            "But I want to stop because I have asthma and even occasional smoking makes it worse.",
            "Maybe I can just drink mocktails at events or be more intentional about my drink choice.",
            "That's a good idea. If I'm not drinking, I probably won't want to smoke. I'll try that.",
            "Yes, I think that's manageable. Thank you.",
        ],
    },
    {
        "patient_id": "sim_p10",
        "description": "Young male, cigarette smoker dealing with anxiety",
        "turns": [
            "I've been smoking for 3 years, about 10 cigarettes a day. I have anxiety issues.",
            "I actually started because I thought it calmed me down. Turns out it makes anxiety worse.",
            "I smoke when I feel a panic coming on or when I'm in an overwhelming social situation.",
            "I tried cutting down once but the anxiety got worse and I gave up.",
            "I want to quit because my therapist told me smoking is feeding my anxiety loop.",
            "Maybe I can use the breathing exercises my therapist taught me instead of smoking.",
            "Yes the 4-7-8 breathing really does help in the moment. I just forget to use it.",
            "I'll put a reminder on my phone to try breathing first before reaching for a cigarette.",
        ],
    },
    {
        "patient_id": "sim_p11",
        "description": "Middle-aged male, tried nicotine patches before, stress triggers",
        "turns": [
            "I've smoked a pack a day for 18 years. I'm 42 now.",
            "My dad died of lung cancer last year. I think about it all the time.",
            "Stress is my biggest trigger. Work, finances, everything piles up.",
            "I've tried nicotine patches and Champix before. Champix gave me nightmares so I stopped.",
            "I really want to quit. My dad's death made it very real for me.",
            "Maybe I should talk to my doctor about other medication options instead of Champix.",
            "Yeah, I'll call my doctor this week and ask specifically about other options.",
            "Okay, I'll set that up. One step at a time. Thank you.",
        ],
    },
    {
        "patient_id": "sim_p12",
        "description": "Young female, social smoker, wants to quit for partner",
        "turns": [
            "I smoke maybe 5 cigarettes a day, been doing it for about 2 years.",
            "My boyfriend doesn't smoke and he has asked me to quit. We're getting serious.",
            "I mainly smoke during work breaks with colleagues. It's a social thing mostly.",
            "I tried quitting cold turkey once for a week. I was very irritable and gave up.",
            "I want to quit for my relationship and honestly also for my own self-image.",
            "Maybe nicotine lozenges could help with the work break cravings without having to smoke.",
            "I'll pick up some lozenges tomorrow and try them at my usual break times.",
            "That sounds like a plan. Thank you for the support.",
        ],
    },
    {
        "patient_id": "sim_p13",
        "description": "Elderly male, 40+ year smoker, acting on doctor's strict advice",
        "turns": [
            "I have been smoking for 42 years. Two packs a day. I am 65 now.",
            "My doctor told me if I don't quit, I am looking at serious heart problems. He was very direct.",
            "I smoke all day. After waking up, after tea, after every meal, throughout work.",
            "I've never seriously tried to quit before. At my age I thought what's the point.",
            "But my doctor frightened me. And my wife is begging me. I want to try for them.",
            "I think I need medical help for this. On my own after 42 years it's too hard.",
            "I'll ask my doctor about nicotine replacement therapy and take it from there.",
            "Okay. I will call the clinic today. Thank you.",
        ],
    },
    {
        "patient_id": "sim_p14",
        "description": "Young male, athlete who smokes secretly, performance suffering",
        "turns": [
            "I'm 22, I play club cricket and I smoke about 8 cigarettes a day but nobody knows.",
            "My stamina has dropped a lot. I get breathless during long fielding sessions.",
            "I smoke after practice when everyone else has gone. It's a secret habit.",
            "I haven't tried quitting. I keep thinking I'll do it after the season.",
            "I want to quit so my performance improves. I want to make the state team next year.",
            "Cold turkey might work for me since I'm young and haven't smoked too long.",
            "Yeah, I think I can go cold turkey and channel the discomfort into training motivation.",
            "I'll stop from tomorrow. Match next week is my goal to be smoke-free by.",
        ],
    },
    {
        "patient_id": "sim_p15",
        "description": "Middle-aged female, WFH smoker, procrastination and cabin fever triggers",
        "turns": [
            "I work from home and I smoke about 12 cigarettes a day spread throughout the day.",
            "I started working from home 3 years ago and my smoking went up a lot after that.",
            "I smoke when I'm procrastinating or when I need a mental break. Cabin fever basically.",
            "I tried tracking my cigarettes once but it depressed me to see the number and I stopped.",
            "I want to quit because I can't keep doing this. My kids are at home too now.",
            "Maybe I can structure real breaks with a short walk outside instead of a cigarette.",
            "I'll set a timer for a 5-minute walk every 90 minutes. That could replace the cigarette breaks.",
            "Okay I'm committing to this. Starting today. Thank you.",
        ],
    },
    {
        "patient_id": "sim_p16",
        "description": "Young male, recently divorced, coping through smoking",
        "turns": [
            "My divorce was finalized 6 months ago and my smoking went from 5 to 20 a day since then.",
            "I'm 33. I used to barely smoke but now I can't stop. It's how I deal with the loneliness.",
            "Evenings are the worst. I'm alone in my apartment and I just smoke through the whole night.",
            "I haven't tried quitting. I don't think I'm ready but I know it's becoming a problem.",
            "I guess I want to cut back. Going back to 5 a day feels more realistic than quitting.",
            "Maybe I need to address the loneliness first. Keeping busy in the evenings.",
            "I'll sign up for an evening gym session. That might kill two birds with one stone.",
            "Yeah, that's actually a good idea. I'll check the schedule today.",
        ],
    },
    {
        "patient_id": "sim_p17",
        "description": "Middle-aged male, financial stress, smoking cheap cigarettes",
        "turns": [
            "I smoke about 15 cheap cigarettes a day. Been doing it for 12 years.",
            "Money is very tight. I'm spending around 2000 rupees a month on cigarettes.",
            "I smoke when I'm stressed about bills or the kids' school fees. Financially it's very hard.",
            "I tried stopping once but my wife was also stressed and when we argued I just went back.",
            "If I quit I save money directly. That's my main reason. It hurts our budget.",
            "Cold turkey might be my only real option since NRT products cost money too.",
            "I'll try going cold turkey and put the daily cigarette money into a jar to see it add up.",
            "That visual of saving will help me. I'll start tomorrow.",
        ],
    },
    {
        "patient_id": "sim_p18",
        "description": "Young female, fitness-conscious, ashamed of smoking habit",
        "turns": [
            "I go to the gym 5 days a week but I smoke 6 to 7 cigarettes a day. I know it's contradictory.",
            "I've been smoking for about 18 months. I hide it from everyone at the gym.",
            "I smoke after workouts weirdly. Also late at night when I'm watching something.",
            "I've tried e-cigarettes as a replacement but I don't think that's actually better.",
            "I want to quit because it's completely at odds with how I see myself. I hate that I do this.",
            "Maybe the post-workout smoke is key to address first since I'm already in a healthy mindset then.",
            "I'll replace the post-workout cigarette with a protein shake and sit with the feeling.",
            "Yes. I'll do that starting from my next session tomorrow morning. Thank you.",
        ],
    },
    {
        "patient_id": "sim_p19",
        "description": "Elderly female, widow, smoking for loneliness and grief",
        "turns": [
            "I have been smoking for 25 years. About 10 cigarettes a day. My husband used to smoke too.",
            "He passed away two years ago. Smoking feels like a connection to him somehow.",
            "Evening time when I am alone in the house is very difficult. That is when I smoke the most.",
            "I haven't tried to stop. I didn't see a reason before.",
            "My children are worried. And I've been having more coughing lately. Maybe it is time.",
            "Perhaps if I have something meaningful to do in the evenings, like calling my grandchildren.",
            "Yes, a regular evening call with my grandchildren would give me something to look forward to.",
            "That is a beautiful idea. I will call them today and make it a routine. Thank you.",
        ],
    },
    {
        "patient_id": "sim_p20",
        "description": "Middle-aged male, construction worker, smokes with coworkers",
        "turns": [
            "I'm a construction site supervisor. I smoke about 20 cigarettes a day for 10 years.",
            "Everyone on site smokes. It's how we take breaks together. Refusing feels antisocial.",
            "Break time is the main trigger. Also early morning before the shift starts.",
            "I tried nicotine gum once but I didn't like it. Felt stupid chewing it on site.",
            "I want to cut down at least. My wife had a baby 3 months ago and I don't want to smoke around the child.",
            "Maybe I can participate in the break without smoking. Just stand there with tea or water.",
            "Yeah I can still be part of the group without smoking. I'll just hold a cup of tea.",
            "That's a plan. I'll try it tomorrow on site. Thank you.",
        ],
    },
]


# ------------------------------------------------------------------
# Main simulation loop
# ------------------------------------------------------------------

def run_simulation():
    total = len(PATIENT_SCENARIOS)
    print(f"Starting simulation: {total} sessions, session_id={SESSION_ID}\n")
    print("=" * 60)

    for i, scenario in enumerate(PATIENT_SCENARIOS, 1):
        patient_id = scenario["patient_id"]
        description = scenario["description"]
        turns = scenario["turns"]

        print(f"\n[{i}/{total}] Patient: {patient_id}")
        print(f"  Context: {description}")
        print(f"  Turns  : {len(turns)}")
        print("-" * 40)

        # Clean slate for each new patient
        reset_for_new_patient(patient_id)

        for t, patient_msg in enumerate(turns, 1):
            print(f"  Turn {t} — Patient: {patient_msg[:70]}{'...' if len(patient_msg) > 70 else ''}")
            try:
                response = therapy_chat(patient_msg, patient_id=patient_id, session_id=SESSION_ID)
                short = response[:100].replace("\n", " ")
                print(f"          Therapist: {short}{'...' if len(response) > 100 else ''}")
            except Exception as e:
                print(f"  [ERROR] Turn {t} failed: {e}")

            # Small delay to avoid rate limiting on the LLM API
            time.sleep(0.5)

        print(f"  -> Log saved: logs/{patient_id}_{SESSION_ID}.json")

    print("\n" + "=" * 60)
    print(f"Simulation complete. {total} sessions logged to logs/")
    print(f"Run evaluation with: python evaluation/evaluate.py --variant full --logs_dir logs")


if __name__ == "__main__":
    run_simulation()

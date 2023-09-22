Title: LLMs for Personalizing Traffic School Learning
Date: 2023-09-22
Category: Generative AI
Slug: llms-for-personalizing-traffic-school-learning
Modified: 2023-09-22
Tags: llm, generative-ai, traffic-school, e-learning
Authors: Saqibur Rahman,
Summary: A soft introduction to a problem we're solving using Generative AI for a customer
<!-- Series: Safe Trafikskola Case Study -->

# Using generative AI to automate a part of Safe Trafikskola's digitized driving license journey

## Problem Statement
Among the many things that we've worked on with Safe Trafikskola as part of their goal to digitize a student's driving license journey, a few key aspects have a lot to be desired.

* Content: Right now, from the app, you can read quite a bit of theory content that's relevant to your 'theory' exam - sure you can "read" but if a student has trouble with any of the material, the next bet is to chat directly with your asssigned mentor. This doesn't really "scale" from a human resource perspective, and there's no limit to the different questions a student can ask.
* Quizzes and Exams: To strengthen your theory knowledge, we have quizzes and exams in the app which follow the MCQ format - while the questions are randomized each time you attempt a quiz, you don't really get any feedback for wrong answers. Further, even if you're weak at a topic, you can't really "practice" it per-se, as you can easily just memorize the answers, or not be relevant to the material you're particularly struggling with.
* Simulator: We have missions in our simulator that students can try right from their homes to "practise" driving without actually booking a practical lesson at the school - ideally, we'd be able to suggest students the missions they need to try out based on a few things: 1. what theory material they're struggling with and 2. what they're struggling with in practical lessons and 3. what they're struggling with in the simulator itself.

Each of these problems have some form of technical solution that we can probably map-out with quite a bit of programming logic. But, what if it's cost efficient to abstract this to an LLM layer that *really* adapts to a student's education.

So, to summarize, combine the different aspects a student experience in their digitized driving license joureny at Safe Trafikskola, and offer a tailored "AI-mentor" who's able to reduce the amount of effort and cost needed for someone to actually get their license. We've built a rich eco-system afterall, so it'll be great to be able to use it.

## Technicalities
Let's start off with a baseline, everything presented today is experimental and proof-of-concept. Since we're dealing with "generative" AI, it'll be hard to gauge a deterministic output/outcome for each step we automate. Regardless, let's shoot for the stars, and maybe we'll land on the moon.


### Content
The theory material we have at Safe Trafikskola right now, is mostly pretty-much static - that is to say, it's been an year and we're just starting to make some new changes. For our demonstration today - let's take a stab at creating a "query-able" knowledge base.


#### Knowledge Base
We have a dump of all the theory content that's accessible to students, organized into chapters. Without going into the nitty-gritty details of [how to build a knowledge base](https://terminusdb.com/blog/vector-database-and-vector-embeddings/), let's summarize as:
* Fetch text data from a source (dump or from a `contents` API
* Create text vector embeddings from text data using any embedding model, we'll use [OpenAI's embedding model](https://platform.openai.com/docs/guides/embeddings/embeddings) in this demo
* Store the vectors in a [vector database](https://www.pinecone.io/learn/vector-database/), we'll use [ChromaDB](https://www.trychroma.com/) for this demo because it's super easy to run and test in a Jupyter Notebook environment

Side-note regarding dynamic content: It's rare that traffic rules, regulations and the theory content designed around will change. Either way, content *can* change. But "what if?" - in which case, storing that content as an embedding in the vector database, when it does change is enough to get the ball rolling.

##### Building the knowledge base
* To keep things simple and also demo how to build your "knowledge base" from any HTML file, I'm going to Safe Trafikskola's content APIs.
* The LangChain module we're going to use is a [document loader](https://python.langchain.com/docs/modules/data_connection/document_loaders/) - more specifically, a `AsyncHtmlLoader` which will let us hit a bunch of web pages and fetch their HTML data.

##### Chapters
We're going to fetch 5 chapters. In order, they discuss:

1. About "traffic" - introduces the student about the pedestrians, cars, and everything on the road that coutns as traffic.
2. About "the road" - what classifies as a "road" and what the lanes are
3. Almost everything about a "B-driving license"
4. About stopping distance and safely stopping the car
5. About the priority of right rule and exiting a junction


```python
from langchain.document_loaders import AsyncHtmlLoader

# URLs for the first 'content' chapters that we have in the system.
urls = [
    # Replace with actualy URLs
    "***",
    "***",
    "***",
    "***",
    "***",
]
html_loader = AsyncHtmlLoader(urls)
content_chapters = html_loader.load()

f"Number of chapters fetched {len(content_chapters)}"

Fetching pages: 100%|#############################################################################| 5/5 [00:02<00:00,  1.93it/s]

'Number of chapters fetched 5'
```


From there, we're going to clean up the HTML a bit, that is, convert it from "raw" HTML with all the tags, and instead into some markdown, something the LLM will understand quite nicely.

Additionally, we're also going to "split" each chapter based on headers, so that topics are clearly separated between each other. This also plays nicely into how many [tokens](https://platform.openai.com/docs/introduction/tokens) you can pass to the model as context.


```python
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import MarkdownTextSplitter

# This nicely converts our HTML to markdown
html2text = Html2TextTransformer()
markdown_documents = html2text.transform_documents(content_chapters)

text_splitter = MarkdownTextSplitter()
content = text_splitter.split_documents(markdown_documents)
```

A small part of what we ended up with:

```python
content[0].page_content[1405:1950] + "..."

'En del fordon med elmotor räknas\nsom cyklar medan andra inte gör det och då många populära fordon i samhället\när eldrivna idag så gäller det att vara uppmärksam på vad det är slags fordon\nsom du framför och om detta är något som du måste ha ett körkort för eller\ninte. Det finns även många lagar som definierar var somliga fordon får och\ninte får framföras samt vilken typ av skyddsutrustning som krävs för att\nframföra dem och beroende på ditt fordonsval så är detta något du måste känna\ntill och har en skyldighet att ta reda på. Motordrivna f...'
```

Now, there's quite a lot we can do when "building" our knowledge base. There are also quite a few caveats and things to consider. Nevertheless, this is a demonstration. So let's move on ahead as this is a good starting point.

Next, we'll create vector embeddings from our text data


```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# This creates embeddings from the text data and stores it in our vector database.
db = Chroma.from_documents(content, OpenAIEmbeddings())
```

So, as an example, let's "ask" our knowledge base about "Trafikanter" a.k.a "Traffic"


```python
# The query is also converted to a vector and then searched against our knowledge-base -
# https://python.langchain.com/docs/integrations/vectorstores/chroma

# English: 'What's a part of traffic?'
answer_context = db.similarity_search("Vad är en del av trafiken?")[0] # We're only going consider the "best" match
answer_context.page_content[:100] + "..."
```




    '# Trafikanter I trafiken är alla “Trafikanter.” Detta gäller vare sig du\npromenerar, kör bil eller r...'



We've correctly retrived the correct context. Now we can "prompt" the model to "generate" an answer based from our "knowledge base".

#### Pretending to be a mentor


```python
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Not the best prompt ever, but again, something that gets the job done.
template = (
    "In this scenario, you're a mentor at the traffic school 'Safe Trafikskola,' "
    "which specializes in teaching students about driving and traffic. You're currently "
    "helping a student prepare for their upcoming driving test. The student is curious "
    "about {context}. Craft a response based on this context, ensuring that your reply includes a leading question "
    "related to the chosen topic. If the user responds to your question, evaluate the accuracy "
    "of their answer based on the context, tell them whether they're right or wrong and provide a short "
    "feedback within one or two sentences in the same language as the user. If the user does not answer "
    "your question, ask them to answer it."
)

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
```


```python
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=chat_prompt,
    output_parser=StrOutputParser()
)

model_answer = chain.run({ "context": answer_context, "text": "Vad är stoppsträcka?" })
model_answer
```




    'Stoppsträcka är den sträcka som ett fordon behöver för att stanna helt efter att föraren har tryckt på bromsen. Det inkluderar både reaktionstiden för föraren att reagera och bromssträckan som fordonet behöver för att stanna. Vad tror du är viktigt att tänka på när det gäller stoppsträcka?'



Translation: "Stopping distance is the distance a vehicle needs to come to a complete stop after the driver has applied the brake. It includes both the reaction time for the driver to react and the braking distance the vehicle needs to stop. What do you think is important to consider when it comes to stopping distance?"

Not bad. If we want, we can pass this as part of the next context, so that the model can try validating whether the student answers this question correctly. But that's a separate problem altogether. Either way, here's a shot.


```python
from langchain.prompts.chat import SystemMessage

chain.prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt, SystemMessage(content=model_answer)])
chain.run({ "context": answer_context, "text": "Reaktionssträckan – beräkning" })
```




    'Viktiga faktorer att tänka på när det gäller stoppsträcka är hastigheten, väglaget och bromssystemets skick. Hastigheten påverkar både reaktionstiden och bromssträckan. Vid högre hastigheter ökar både reaktionstiden och bromssträckan, vilket leder till en längre stoppsträcka. Väglaget, såsom våt eller isig vägbana, kan också påverka bromssträckan. Slutligen är det viktigt att ha ett fungerande bromssystem för att kunna stanna på kortast möjliga sträcka.'



Not the best answer, but not the worst either. There are couple of ways to "fix" this:
* Fine-tuning the model to act the way we want it to by feeding a bunch of QnA examples where it leads to students to an answer - https://platform.openai.com/docs/guides/fine-tuning
* Better prompts

### Quizzes and Exams
Moving on to the next phase, quizzes. Let's try to:
* Generate new questions based on the "context" that we can show in the app
  


```python
# We're going to set-up it, so that the JSON format is something the mobile-app already expects.
# This also (in most cases) ensures that the app doesn't break with a malformed response

from typing import List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator


class Answer(BaseModel):
    text: str
    is_correct: bool

class Question(BaseModel):
    uuid: str
    text: str
    answers: List[Answer]

class Quiz(BaseModel):
    questions: List[Question]
```


```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

question = "Jag vill träna på B-körkort."
answer_context = db.similarity_search(question)[0] # We're only going consider the "best" match
answer_context.page_content[:100] + "..."

parser = PydanticOutputParser(pydantic_object=Quiz)
format_instructions = parser.get_format_instructions()

quiz_template = (
    "Based on the context = {context}, create 5 different multiple choice questions with a single right answer using a question as the title "
    ", in {format_instructions} "
    "in the same language as the context."
)


prompt = PromptTemplate(
    template=quiz_template,
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(context=answer_context)

model_name = "gpt-3.5-turbo"
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)
output = model(_input.to_string())

quiz = parser.parse(output)
```

    Quiz(questions=[Question(uuid='1', text='Vad innebär ett B-körkort?', answers=[Answer(text='Rättigheter att köra en massa saker', is_correct=True), Answer(text='Endast rättigheter att köra en vanlig bil', is_correct=False), Answer(text='Rättigheter att köra buss med över åtta sittplatser', is_correct=False), Answer(text='Endast rättigheter att köra terränghjuling', is_correct=False)]), Question(uuid='2', text='Vilken typ av fordon får man inte köra med ett B-körkort?', answers=[Answer(text='Personbil med högst 8 sittplatser', is_correct=False), Answer(text='Buss med över åtta sittplatser', is_correct=True), Answer(text='Lätt släpfordon med totalvikt över 750kg', is_correct=False), Answer(text='Motorredskap klass 1', is_correct=False)]), Question(uuid='3', text='Vad krävs för att få köra terränghjuling?', answers=[Answer(text='Ett förarbevis för terränghjuling', is_correct=True), Answer(text='Ett B-körkort', is_correct=False), Answer(text='Ett förarbevis för motorredskap klass 1', is_correct=False), Answer(text='Ett förarbevis för terrängvagn', is_correct=False)]), Question(uuid='4', text='Vilken typ av fordon får man köra med ett B-körkort?', answers=[Answer(text='Moped klass 2', is_correct=True), Answer(text='Terrängvagn', is_correct=False), Answer(text='Fyrhjulig motorcykel', is_correct=False), Answer(text='Lastbilssläp', is_correct=False)]), Question(uuid='5', text='Vad ingår i behörigheten för ett B-körkort?', answers=[Answer(text='Moped klass 1', is_correct=True), Answer(text='Motorredskap klass 2', is_correct=True), Answer(text='A och B traktorer', is_correct=True), Answer(text='Terränghjuling', is_correct=False)])])


    {'questions': [{'uuid': '1',
       'text': 'Vad innebär ett B-körkort?',
       'answers': [{'text': 'Rättigheter att köra en massa saker',
         'is_correct': True},
        {'text': 'Endast rättigheter att köra en vanlig bil', 'is_correct': False},
        {'text': 'Rättigheter att köra buss med över åtta sittplatser',
         'is_correct': False},
        {'text': 'Endast rättigheter att köra terränghjuling',
         'is_correct': False}]},
      {'uuid': '2',
       'text': 'Vilken typ av fordon får man inte köra med ett B-körkort?',
       'answers': [{'text': 'Personbil med högst 8 sittplatser',
         'is_correct': False},
        {'text': 'Buss med över åtta sittplatser', 'is_correct': True},
        {'text': 'Lätt släpfordon med totalvikt över 750kg', 'is_correct': False},
        {'text': 'Motorredskap klass 1', 'is_correct': False}]},
      {'uuid': '3',
       'text': 'Vad krävs för att få köra terränghjuling?',
       'answers': [{'text': 'Ett förarbevis för terränghjuling',
         'is_correct': True},
        {'text': 'Ett B-körkort', 'is_correct': False},
        {'text': 'Ett förarbevis för motorredskap klass 1', 'is_correct': False},
        {'text': 'Ett förarbevis för terrängvagn', 'is_correct': False}]},
      {'uuid': '4',
       'text': 'Vilken typ av fordon får man köra med ett B-körkort?',
       'answers': [{'text': 'Moped klass 2', 'is_correct': True},
        {'text': 'Terrängvagn', 'is_correct': False},
        {'text': 'Fyrhjulig motorcykel', 'is_correct': False},
        {'text': 'Lastbilssläp', 'is_correct': False}]},
      {'uuid': '5',
       'text': 'Vad ingår i behörigheten för ett B-körkort?',
       'answers': [{'text': 'Moped klass 1', 'is_correct': True},
        {'text': 'Motorredskap klass 2', 'is_correct': True},
        {'text': 'A och B traktorer', 'is_correct': True},
        {'text': 'Terränghjuling', 'is_correct': False}]}]}



While it did mess up the UUID, this is still quite good. Here's an English translation:

```
{'questions': [{'uuid': '1',
   'text': 'What does a B driver's license entail?',
   'answers': [{'text': 'Rights to drive a variety of things',
     'is_correct': True},
    {'text': 'Only rights to drive a regular car', 'is_correct': False},
    {'text': 'Rights to drive a bus with over eight seats',
     'is_correct': False},
    {'text': 'Only rights to drive an all-terrain vehicle',
     'is_correct': False}]},
  {'uuid': '2',
   'text': 'What type of vehicle are you not allowed to drive with a B driver's license?',
   'answers': [{'text': 'Passenger car with a maximum of 8 seats',
     'is_correct': False},
    {'text': 'Bus with over eight seats', 'is_correct': True},
    {'text': 'Light trailer with a total weight over 750kg',
     'is_correct': False},
    {'text': 'Motor implement class 1', 'is_correct': False}]},
  {'uuid': '3',
   'text': 'What is required to drive an all-terrain vehicle?',
   'answers': [{'text': 'A driver's permit for all-terrain vehicles',
     'is_correct': True},
    {'text': 'A B driver's license', 'is_correct': False},
    {'text': 'A driver's permit for motor implements class 1',
     'is_correct': False},
    {'text': 'A driver's permit for off-road vehicles', 'is_correct': False}]},
  {'uuid': '4',
   'text': 'What type of vehicle can you drive with a B driver's license?',
   'answers': [{'text': 'Moped class 2', 'is_correct': True},
    {'text': 'Off-road vehicle', 'is_correct': False},
    {'text': 'Quad bike', 'is_correct': False},
    {'text': 'Truck trailer', 'is_correct': False}]},
  {'uuid': '5',
   'text': 'What is included in the entitlement of a B driver's license?',
   'answers': [{'text': 'Moped class 1', 'is_correct': True},
    {'text': 'Motor implement class 2', 'is_correct': True},
    {'text': 'A and B tractors', 'is_correct': True},
    {'text': 'All-terrain vehicle', 'is_correct': False}]}]}
```

One of the challenges we had during development is giving feedback to the student when they answered incorrectly. The teachers at the school didn't really have the time to go back and add customized feedback for the wrong answers. Let's see what the model can do. Assuming the following JSON input (limited to 2 questions):


```python
quiz_attempt = {
  "questions": [
    {
      "uuid": "1",
      "text": "What does a B driver's license entail?",
      "answers": [
        {
          "text": "Rights to drive a variety of things",
          "is_correct": True,
          "is_selected": False 
        },
        {
          "text": "Only rights to drive a regular car",
          "is_correct": False,
          "is_selected": False
        },
        {
          "text": "Rights to drive a bus with over eight seats",
          "is_correct": False,
          "is_selected": False
        },
        {
          "text": "Only rights to drive an all-terrain vehicle",
          "is_correct": False,
          "is_selected": False
        }
      ]
    },
    {
      "uuid": "2",
      "text": "What type of vehicle are you not allowed to drive with a B driver's license?",
      "answers": [
        {
          "text": "Passenger car with a maximum of 8 seats",
          "is_correct": False,
          "is_selected": False
        },
        {
          "text": "Bus with over eight seats",
          "is_correct": True,
          "is_selected": False
        },
        {
          "text": "Light trailer with a total weight over 750kg",
          "is_correct": False,
          "is_selected": False
        },
        {
          "text": "Motor implement class 1",
          "is_correct": False,
          "is_selected": False
        }
      ]
    }
  ]
}
```


```python
template = (
    "A user has just submitted a quiz - {context}. Go through each question, validate whether the answer is correct based "
    "based on whether correct answers were selected, and state whether the user is correct or incorrect, and add an explanation "
    "with feedback so that the user knows why they're right or wrong. Add your feedback 'feedback' to each questions. "
    "Your output should be JSON as well."
)

chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(template)]),
    output_parser=StrOutputParser()
)

chain.run({ "context": quiz_attempt })
```




    '{"questions": [\n    {\n        "uuid": "1",\n        "text": "What does a B driver\'s license entail?",\n        "answers": [\n            {"text": "Rights to drive a variety of things", "is_correct": true, "is_selected": false},\n            {"text": "Only rights to drive a regular car", "is_correct": false, "is_selected": false},\n            {"text": "Rights to drive a bus with over eight seats", "is_correct": false, "is_selected": false},\n            {"text": "Only rights to drive an all-terrain vehicle", "is_correct": false, "is_selected": false}\n        ],\n        "feedback": "The correct answer is \'Rights to drive a variety of things\'. A B driver\'s license allows you to drive a regular car as well as other vehicles such as motorcycles and light trucks."\n    },\n    {\n        "uuid": "2",\n        "text": "What type of vehicle are you not allowed to drive with a B driver\'s license?",\n        "answers": [\n            {"text": "Passenger car with a maximum of 8 seats", "is_correct": false, "is_selected": false},\n            {"text": "Bus with over eight seats", "is_correct": true, "is_selected": false},\n            {"text": "Light trailer with a total weight over 750kg", "is_correct": false, "is_selected": false},\n            {"text": "Motor implement class 1", "is_correct": false, "is_selected": false}\n        ],\n        "feedback": "The correct answer is \'Bus with over eight seats\'. With a B driver\'s license, you are not allowed to drive a bus with over eight seats."\n    }\n]}'



Converting the answer to a more readable JSON format, we get, with the fluff removed:

```
{
  "questions": [
    {
      "text": "What does a B driver's license entail?",
      "feedback": "The correct answer is 'Rights to drive a variety of things'. A B driver's license allows you to drive a regular car as well as other vehicles such as motorcycles and light trucks."
    },
    {
      "text": "What type of vehicle are you not allowed to drive with a B driver's license?",
      "feedback": "The correct answer is 'Bus with over eight seats'. With a B driver's license, you are not allowed to drive a bus with over eight seats."
    }
  ]
}

```

Honestly, not too bad. I'd like to see how it does with more complicated "data" but not a bad start for a PoC.

In summary; we're able to (as a PoC):

* Generate new questions based on pre-defined content
* Evaluate and give feedback to the student based on their quiz attempts

### Simulator
Now for the most exciting part - guiding a student towards a specific mission that'll help them address their driving weaknesses.

Let's paint the scenario a little bit.

* You have access to the first 60 missions we have in the simulator right now: `https://***/v1/missions/?type=automatic`
* You're going to practice for your automatic B-driving license
* You're struggling with roundabouts - well we have a mission for that.
* Let's see if the "AI-mentor" can guide the student to "practice" that mission.


From a technical perspective - we should:
* Fetch all the information about the missions
* Store them in our vector database as embeddings
* Query the vector database and return all "relevant" matches
* Pass the matches to model and give the user what they want


```python
# For the demo, I've saved the mission information in a local .json file

missions = "./missions.json"

from langchain.document_loaders import JSONLoader
from pprint import pprint

loader = JSONLoader(file_path=missions, jq_schema='.automatic_missions.sections[].subsections[].missions[] | "\(.title): \(.description)"')
data = loader.load()

db = Chroma.from_documents(data, OpenAIEmbeddings())
```


```python
user_query = "I can't drive around roundabouts. What can I do?"
```


```python
mission_context = db.similarity_search(user_query)
```


```python
template = (
    "As a traffic school mentor, suggest what the user can do to deal with what they're struggling with - {query}"
    "Based on the context - {context}, if the user needs to practice a mission, suggest which missions a user should practice. "
    "Use the sequence numbers in the meta data as the mission numbers in your output. "
    "Also mention why you're suggesting that mission."
)

chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(template)]),
    output_parser=StrOutputParser()
)

pprint(chain.run({ "context": mission_context, "query": user_query }))
```

    ("If you're struggling with driving around roundabouts, I suggest practicing "
     'Mission 42 and Mission 43. \n'
     '\n'
     'Mission 42 involves driving through three roundabouts with traffic and '
     'making a right turn at the exit. This mission will help you gain experience '
     'in navigating roundabouts with other vehicles present, which can be a common '
     'challenge for many drivers.\n'
     '\n'
     'Mission 43, on the other hand, involves driving through three roundabouts '
     'without any traffic and making a simple right turn. This mission will allow '
     'you to practice navigating roundabouts in a less stressful environment, '
     'allowing you to focus on the fundamental techniques and build your '
     'confidence.\n'
     '\n'
     'By practicing these missions, you will be able to familiarize yourself with '
     'the roundabout driving techniques and gain the necessary skills to drive '
     'around roundabouts with ease. Remember to always follow the traffic rules '
     'and signals while driving.')


Not too bad given that the model only had desriptions of the missions to go off of. With a bit more metadata, we can turn it into a richer experience.

### Next steps
The next thing to work on is connecting all of this via "chains" (it's LangChain
afterall), where the model feeds outputs from one stage into another and also
takes decisions to truly "mentor" the student based on what they're asking.

* Chains: [https://python.langchain.com/docs/modules/chains/](https://python.langchain.com/docs/modules/chains/)
* Agents: [https://python.langchain.com/docs/modules/agents.html](https://python.langchain.com/docs/modules/agents.html)

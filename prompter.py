"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
import pdb
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class PrompterESC(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
        # self.selections = ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)"]
        # self.selections = ["1", "2", "3", "4", "5", "6", "7", "8"]
        self.strategies = ["Question", "Restatement or Paraphrasing", "Reflection of feelings", "Self-disclosure",
                           "Affirmation and Reassurance", "Providing Suggestions or Information", "Greeting", "Others"]
        self.strategy_inf = {
            "Question": "This emotion support strategy involves inquiring about details concerning the problem to aid the individual seeking help in expressing their specific challenges. Open-ended questions are highly effective in this regard, while closed questions can be employed to obtain more precise information.",
            "Restatement or Paraphrasing": "Restating or paraphrasing refers to the act of succinctly rephrasing the help-seeker's statements, which can assist them in gaining a clearer perspective on their situation.",
            "Reflection of feelings": "The technique of reflecting feelings involves effectively expressing and describing the emotions experienced by the individual seeking help.",
            "Self-disclosure": "Self-disclosure entails sharing relevant personal experiences or emotions that resonate with the help-seeker, thus demonstrating empathy.",
            "Affirmation and Reassurance": "Affirmation and reassurance within the emotion support strategy involve acknowledging and validating the help-seeker's strengths, motivation, and abilities, while also offering reassurance and encouragement.",
            "Providing Suggestions or Information": "The aspect of providing suggestions or information within the emotion support strategy involves offering recommendations on potential changes, while being cautious not to dictate or prescribe specific actions. It also encompasses providing helpful information to the help-seeker, such as data, facts, opinions, resources, or responding to their inquiries.",
            "Greeting": "Greeting is about exchange pleasantries.",
            "Others": "Use other support strategies that do not fall into the above categories.",

            # "Question": "Asking for information related to the problem to help the help-seeker articulate the issues that they face. Open-ended questions are best, and closed questions can be used to get speciﬁc information.",
            # "Restatement or Paraphrasing": "A simple, more concise rephrasing of the help-seeker’s statements that could help them see their situation more clearly.",
            # "Reflection of feelings": "Articulate and describe the help-seeker’s feelings.",
            # "Self-disclosure": "Divulge similar experiences that you have had or emotions that you share with the help-seeker to express your empathy.",
            # "Affirmation and Reassurance": "Affirm the helpseeker’s strengths, motivation, and capabilities and provide reassurance and encouragement.",
            # "Providing Suggestions or Information": "Provide suggestions about how to change, but be careful to not overstep and tell them what to do, or provide useful information to the help-seeker, for example with data, facts, opinions, resources, or by answering questions.",
            # "Greeting": "Exchange pleasantries.",
            # "Others": "Use other support strategies that do not fall into the above categories."
        }

    def generate_prompt(self, data_point, test=False, response=False):
        prompt = ""
        output = ""
        data_input = ""
        data = {}

        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break
        data_point['dialog'] = data_point['dialog'][:j + 1]
        if response:
            '''There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress:

                {input}
                Up to now, the emotion supporter need to response properly to the seeker to make them feel better. 
                Before responding, the supporter should follow some strategies to standardize the response generation.If you are the supporter, according to the conversation, please choose a proper strategy first to respond according to the context.

                The picked strategy: {strategy} 

                Now complete the conversation based on the picked strategy. The response of the conversation: {output}'''

            '''Up to now, the emotion supporter need to response properly to the seeker to make them feel better.
Before responding, the supporter should follow some strategies to standardize the response generation. If you are the supporter, according to the conversation, please choose a proper strategy first to respond according to the context.'''
            prompt += \
'''There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress:

{input}

Up to now, the emotion supporter need to response properly to the seeker to make them feel better.
Before responding, the supporter should follow some strategies to standardize the response generation. If you are the supporter, according to the conversation, please choose a proper strategy first from candidate strategies listed below:

'''
            # for i in range(len(self.strategies)):
            #     # prompt += "{} {}".format(self.selections[i], self.strategies[i]) + " "  # + "({})\n".format(self.strategy_inf[self.strategies[i]])
            #     prompt += "{} {}".format(self.selections[i], self.strategies[i]) + " ({})\n".format(
            #         self.strategy_inf[self.strategies[i]])
            #     # input
            # prompt += "\n"
            for j in range(len(data_point['dialog']) - 1, -1, -1):
                if data_point['dialog'][j]['speaker'] == 'usr':
                    continue
                break

            data_point['dialog'] = data_point['dialog'][:j + 1]
            import random
            random.seed(random.randint(1, 1000))
            # random.shuffle(self.strategies)
            for i, dia in enumerate(data_point['dialog']):
                if i == len(data_point['dialog']) - 1:
                    # strategy = dia['strategy']
                    # strategy = dia['strategy']

                    # strategy = self.strategies[random.randint(0,len(self.strategies)-1)]
                    # strategy = 'Greeting'
                    # print(strategy,end=' ')
                    index = random.randint(0, len(self.selections) - 1)
                    strategy = "{} {}".format(self.selections[index], self.strategies[index])
                    output = dia['text']
                    break
                if dia['speaker'] == 'usr':
                    data_input += '({})Help seeker: '.format(i + 1)
                else:
                    data_input += '({})Supporter: '.format(
                        i + 1)  # + 'Strategy chosen: ' + dia['annotation']['strategy']
                data_input += dia['text'].strip('\n') + '\n'

            if test:
                prompt += "The chosen strategy is: {strategy} " + "({})".format(self.strategy_inf[
                                                                                    strategy]) + "\n\nNow complete the conversation based on the picked strategy. The response of the conversation: "
                # prompt += "The chosen strategy is: {strategy} \n\nNow complete the conversation based on the picked strategy. The response of the conversation: "
                data.update({'input': data_input, 'strategy': strategy})
            else:
                prompt += "The chosen strategy is: {strategy} " + "({})".format(self.strategy_inf[
                                                                                    self.strategies[
                                                                                        index]]) + "\n\nNow complete the conversation based on the picked strategy. The response of the conversation: {output}"
                data.update({'input': data_input, 'strategy': strategy, 'output': output})
        else:
            '''There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress:

               {input}
               Up to now, the emotion supporter need to response properly to the seeker to make them feel better.
               Before responding, the supporter should follow some strategies to standardize the response generation. If you are the supporter, according to the conversation, please choose a proper strategy first from candidate strategies listed below:

               (a) Question
               (b) Restatement or Paraphrasing
               (c) Reflection of feelings
               (d) Self-disclosure
               (e) Affirmation and Reassurance
               (f) Providing Suggestions or Information
               (g) Greeting
               (h) Others

               The chosen strategy is:{output}'''
            prompt += \
                '''There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress:
                
                {input}
                Up to now, the emotion supporter need to response properly to the seeker to make them feel better.
                Before responding, the supporter should follow some strategies to standardize the response generation. If you are the supporter, according to the conversation, please choose a proper strategy first from candidate strategies listed below:
                
                '''
            #             prompt +=\
            # '''You are a person skilled in the theory of emotional support. You understand that there are three stages to achieve emotional support: exploration, comfort and action, and you will use the following eight strategies flexibly and choose one strategy to respond according to the context.
            #
            # 1.Question
            # 2.Restatement or Paraphrasing
            # 3.Reflection of feelings
            # 4.Self-disclosure
            # 5.Affirmation and Reassurance
            # 6.Providing Suggestions
            # 7.Information
            # 8.Others
            #
            # The dialogue context is:
            #
            # {input}
            #
            # You should first output the strategy you choose and then generate the response grounding in it.'''
            import random
            random.seed(random.randint(1, 1000))
            random.shuffle(self.strategies)
            for i in range(len(self.strategies)):
                # prompt += "{} {}".format(self.selections[i], self.strategies[i]) + " "  # + "({})\n".format(self.strategy_inf[self.strategies[i]])
                prompt += "{}.{}".format(self.selections[i], self.strategies[i]) + " ({})\n".format(
                    self.strategy_inf[self.strategies[i]])
                # input
            prompt += "\n"
            for j in range(len(data_point['dialog']) - 1, -1, -1):
                if data_point['dialog'][j]['speaker'] == 'usr':
                    continue
                break
            data_point['dialog'] = data_point['dialog'][:j + 1]

            for i, dia in enumerate(data_point['dialog']):
                if i == len(data_point['dialog']) - 1:
                    try:
                        index = self.strategies.index(dia['strategy'])
                    except:
                        pdb.set_trace()
                    output = "{}".format(self.strategies[index])
                    break
                if dia['speaker'] == 'usr':
                    data_input += '({})Help seeker: '.format(i + 1)
                else:
                    data_input += '({})Supporter: '.format(
                        i + 1)  # + 'Strategy chosen: ' + dia['annotation']['strategy']
                data_input += dia['text'].strip('\n') + '\n'
            prompt += 'When selecting a strategy, please take into account the three stages required to achieve emotional support: exploration, comfort, and action. Analyze which stages both the supporter and the help seeker are currently experiencing.  Let\' think step by step.\n'
            # response
            if test:
                prompt += "The chosen strategy is: "
                data.update({'input': data_input})
            else:
                prompt += "The chosen strategy is: {output}"
                data.update({'input': data_input, 'output': output})

        prompt = prompt.format_map(data)
        return prompt, output


class PrompterED(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

        self.EMO_MAP = {
            "surprised": 0,
            "excited": 1,
            "annoyed": 2,
            "proud": 3,
            "angry": 4,
            "sad": 5,
            "grateful": 6,
            "lonely": 7,
            "impressed": 8,
            "afraid": 9,
            "disgusted": 10,
            "confident": 11,
            "terrified": 12,
            "hopeful": 13,
            "anxious": 14,
            "disappointed": 15,
            "joyful": 16,
            "prepared": 17,
            "guilty": 18,
            "furious": 19,
            "nostalgic": 20,
            "jealous": 21,
            "anticipating": 22,
            "embarrassed": 23,
            "content": 24,
            "devastated": 25,
            "sentimental": 26,
            "caring": 27,
            "trusting": 28,
            "ashamed": 29,
            "apprehensive": 30,
            "faithful": 31,
        }

    def generate_prompt(self, data_point, set=False, test=False, response=False, file_path=''):
        prompt = ""
        # concept = "sadness: an emotional pain associated with, or  characterized by, feelings of disadvantage, loss, despair, grief,  helplessness, disappointment and sorrow."
        context = ""
        data = {}
        output = data_point['target'].capitalize()
        # 使用另一个分类lora给出的情绪分类结果作为生成的依据
        if file_path != '':
            emo_file = json.load(open(file_path, 'r', encoding='utf-8'))
            context = data_point['context'][0]
            file_data = json.load(open(file_path, 'r', encoding='utf-8'))
            emotion = data_point['emotion'].capitalize()
            for tst in file_data:
                tst_data = tst[0]
                if context in tst_data:
                    locate_str = 'The emotion of the speaker in the above conversation is :'
                    emotion = tst_data[tst_data.find(locate_str) + len(locate_str):].strip().capitalize()
                    emotion = emotion.replace("<unk>", "")
                    # print(emotion)
                    break
            else:
                print("Not Found!!!..............................")
        else:
            emotion = data_point['emotion'].capitalize()
        locate_str = 'The emotion of the speaker in the above conversation is :'
        model_generated = [sample[0][sample[0].find(locate_str) + len(locate_str):].strip() for sample in data]

        relations = ["oIntent", "oNeed", "oWant", "oEffect", "oReact"]
        last_utt_intent = ' or '.join([' '.join(i) for i in data_point['comet'][-1][0][:3] if i != ['none']]).replace(".", "")
        last_utt_need = ' or '.join([' '.join(i) for i in data_point['comet'][-1][1][:3] if i != ['none']]).replace(".", "")
        last_utt_want = ' or '.join([' '.join(i) for i in data_point['comet'][-1][2][:3] if i != ['none']]).replace(".", "")
        last_utt_effect = ' or '.join([' '.join(i) for i in data_point['comet'][-1][-2][:3] if i != ['none']]).replace(".", "")
        last_utt_react = ' or '.join([' '.join(i) for i in data_point['comet'][-1][-1][:3] if i != ['none']]).replace(".", "")
        comet = 'After posting the last utterance, the listener intent to {}, need to {} and want to {}, the listener may feel {}, additionally the listener would {}.'.format(last_utt_intent, last_utt_need, last_utt_want, last_utt_react,
                                                                                                                                                                               last_utt_effect)

        for index, turn in enumerate(data_point['context']):
            if index % 2 == 0:
                context += '({})Speaker: '.format(index + 1)
            else:
                context += '({})Listener: '.format(index + 1)
            context += turn + "\n"
        # print(comet)
        # pdb.set_trace()
        concept = \
            '''
1. Surprised: Feeling taken aback or amazed by something unexpected.
2. Excited: Feeling a strong sense of enthusiasm or anticipation.
3. Annoyed: Feeling irritated or bothered by someone or something.
4. Proud: Feeling a sense of satisfaction or accomplishment in oneself or someone else.
5. Angry: Feeling intense displeasure or strong emotions of rage.
6. Sad: Feeling unhappy or sorrowful, often accompanied by tears or a sense of loss.
7. Grateful: Feeling thankful and appreciative of something or someone.
8. Lonely: Feeling isolated or lacking companionship.
9. Impressed: Feeling admiration or respect for someone or something.
10. Afraid: Feeling scared or fearful, often in anticipation of potential danger.
11. Disgusted: Feeling strong aversion or revulsion towards something unpleasant.
12. Confident: Feeling self-assured and certain in one's abilities or qualities.
13. Terrified: Feeling extreme fear or terror.
14. Hopeful: Feeling optimistic and having a positive outlook for the future.
15. Anxious: Feeling uneasy or worried about an uncertain or upcoming event.
16. Disappointed: Feeling let down or unsatisfied due to unmet expectations.
17. Joyful: Feeling great happiness or delight.
18. Prepared: Feeling ready or equipped for a particular situation or task.
19. Guilty: Feeling remorse or self-reproach for having done something wrong.
20. Furious: Feeling extreme anger and rage.
21. Nostalgic: Feeling a sentimental longing or affection for the past.
22. Jealous: Feeling envious or resentful towards someone's achievements, possessions, or relationships.
23. Anticipating: Feeling excited or looking forward to something in the future.
24. Embarrassed: Feeling self-conscious, awkward, or ashamed in a social situation.
25. Content: Feeling a state of satisfaction, peace, and overall happiness.
26. Devastated: Feeling overwhelmed with extreme sadness or grief.
27. Sentimental: Feeling nostalgic or emotional, often in response to sentimental memories or objects.
28. Caring: Feeling a deep concern, empathy, or affection towards others.
29. Trusting: Feeling confident in someone's reliability, honesty, or integrity.
30. Ashamed: Feeling embarrassed or guilty due to one's actions or behavior.
31. Apprehensive: Feeling anxious or uneasy about something that may happen in the future.
32. Faithful: Feeling loyal, committed, and steadfast in one's support or allegiance.'''
        if response:
            prompt += \
'''Assuming that you are a highly empathetic person, you should first identify emotion of the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation.

Please indicate the emotion of the speaker in the following conversation. The emotion labels are as follows:
{concept}

The conversation is as follows:
{context}

{comet}

The emotion of the speaker in the above conversation is : {emotion}.
'''
            # '''Assuming that you are a highly empathetic person, you should first identify emotion of the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation.
            #
            # Please indicate the emotion of the speaker in the following conversation. The emotion labels are as follows:
            # {concept}
            #
            # The conversation is as follows:
            # {context}
            #
            # {comet}
            # Before responding, please first identify the emotion of the dyadic dialogue clip based on the knowledge in the above sentence.
            # '''

            if test:
                prompt += \
                    '''Now, please leverage the knowledge provided above to generate a concise, relevant and empathetic response for the following conversation: '''
                data.update({"context": context, 'concept': concept, "emotion": emotion, 'comet': comet})
                # data.update({"context": context, 'concept': concept, 'comet': comet})
            else:
                prompt += \
                    '''Now, please leverage the knowledge provided above to generate a concise, relevant and empathetic response for the following conversation: {output}'''
                data.update({"context": context, 'concept': concept, "emotion": emotion, 'comet': comet, 'output': output})
                # data.update({"context": context, 'concept': concept, 'comet': comet, 'output': output})
            prompt = prompt.format_map(data)
            return prompt, output
        else:
            if set:
                prompt += \
'''Assuming that you are a highly empathetic person, you should first identify emotion of the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation.

Please indicate the emotion of the speaker in the following conversation. The emotion label can only come from the set {{surprised, excited, annoyed, proud, angry, sad, grateful, lonely, impressed, afraid, disgusted, confident, terrified, hopeful, anxious, disappointed, joyful, prepared, guilty, furious, nostalgic, jealous, anticipating, embarrassed, content, devastated, sentimental, caring, trusting, ashamed, apprehensive, faithful}}. The concepts of these labels are as follows:
'''
            else:
                prompt += \
'''Assuming that you are a highly empathetic person, you should first identify emotion of the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation.

Please indicate the emotion of the speaker in the following conversation. The emotion labels are as follows:
'''

            knowledge = \
'''Three subsequent event about the help seeker that happens or could happen following the last the utterance stated by the help seeker: 
{}

Three possible emotional reaction of the help seeker in response to the last utterance stated by the help seeker are: 
{}

Three underlying cause of the last utterance (the reason contributing to the utterance stated by the help seeker) are: 
{}'''.format(data_point['ChatGPT_cause'].replace('\n', ''), data_point['ChatGPT_emo'].replace('\n', ''), data_point['ChatGPT_subs'].replace('\n', ''))
            if test:
                prompt += \
'''{concept}

The conversation is as follows:
{context}

{knowledge}

The emotion of the speaker in the above conversation is :'''
                # data.update({"context": context, "concept": concept, "emotion": emotion, "comet": comet,"knowledge":knowledge})
                data.update({"context": context, "concept": concept, "emotion": emotion, "knowledge": knowledge})
            else:
                prompt += \
'''{concept}

The conversation is as follows:
{context}

{knowledge}

The emotion of the speaker in the above conversation is : {emotion}.'''

                # data.update({"context": context, "concept": concept, "emotion": emotion, "comet": comet, "output": output, "knowledge": knowledge})
            data.update({"context": context, "concept": concept, "emotion": emotion, "output": output, "knowledge": knowledge})
            prompt = prompt.format_map(data)
            return prompt, emotion


class PrompterCCIESC(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
        # self.selections = ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)"]
        # self.selections = ["1", "2", "3", "4", "5", "6", "7", "8"]
        self.strategies = ["Question", "Restatement or Paraphrasing", "Reflection of feelings", "Self-disclosure",
                           "Affirmation and Reassurance", "Providing Suggestions or Information", "Greeting", "Others"]
        self.strategy_inf = {
            "Question": "This emotion support strategy involves inquiring about details concerning the problem to aid the individual seeking help in expressing their specific challenges. Open-ended questions are highly effective in this regard, while closed questions can be employed to obtain more precise information.",
            "Restatement or Paraphrasing": "Restating or paraphrasing refers to the act of succinctly rephrasing the help-seeker's statements, which can assist them in gaining a clearer perspective on their situation.",
            "Reflection of feelings": "The technique of reflecting feelings involves effectively expressing and describing the emotions experienced by the individual seeking help.",
            "Self-disclosure": "Self-disclosure entails sharing relevant personal experiences or emotions that resonate with the help-seeker, thus demonstrating empathy.",
            "Affirmation and Reassurance": "Affirmation and reassurance within the emotion support strategy involve acknowledging and validating the help-seeker's strengths, motivation, and abilities, while also offering reassurance and encouragement.",
            "Providing Suggestions or Information": "The aspect of providing suggestions or information within the emotion support strategy involves offering recommendations on potential changes, while being cautious not to dictate or prescribe specific actions. It also encompasses providing helpful information to the help-seeker, such as data, facts, opinions, resources, or responding to their inquiries.",
            "Greeting": "Greeting is about exchange pleasantries.",
            "Others": "Use other support strategies that do not fall into the above categories.",

            # "Question": "Asking for information related to the problem to help the help-seeker articulate the issues that they face. Open-ended questions are best, and closed questions can be used to get speciﬁc information.",
            # "Restatement or Paraphrasing": "A simple, more concise rephrasing of the help-seeker’s statements that could help them see their situation more clearly.",
            # "Reflection of feelings": "Articulate and describe the help-seeker’s feelings.",
            # "Self-disclosure": "Divulge similar experiences that you have had or emotions that you share with the help-seeker to express your empathy.",
            # "Affirmation and Reassurance": "Affirm the helpseeker’s strengths, motivation, and capabilities and provide reassurance and encouragement.",
            # "Providing Suggestions or Information": "Provide suggestions about how to change, but be careful to not overstep and tell them what to do, or provide useful information to the help-seeker, for example with data, facts, opinions, resources, or by answering questions.",
            # "Greeting": "Exchange pleasantries.",
            # "Others": "Use other support strategies that do not fall into the above categories."
        }

    def extract_inf(self, inf_sentence):
        sentence_list = inf_sentence.split('\n')
        sentence_list = [sen for sen in sentence_list if sen != '' and ('1' in sen or '2' in sen or '3' in sen)]
        result = ' '.join(sentence_list[0].split(' ')[1:])
        pdb.set_trace()

        return result

    def generate_prompt(self, data_point, test=False, response=False):
        prompt = ""
        output = ""
        data_input = ""
        data = {}

        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break
        data_point['dialog'] = data_point['dialog'][:j + 1]
        if response:
            '''There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress:

                {input}
                Up to now, the emotion supporter need to response properly to the seeker to make them feel better. 
                Before responding, the supporter should follow some strategies to standardize the response generation.If you are the supporter, according to the conversation, please choose a proper strategy first to respond according to the context.

                The picked strategy: {strategy} 

                Now complete the conversation based on the picked strategy. The response of the conversation: {output}'''
            '''Up to now, the emotion supporter need to response properly to the seeker to make them feel better.
Before responding, the supporter should follow some strategies to standardize the response generation. If you are the supporter, according to the conversation, please choose a proper strategy first to respond according to the context.'''
            prompt += \
'''There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress:

{input}
Up to now, the emotion supporter need to response properly to the seeker to make them feel better.
Before responding, the supporter should follow some strategies to standardize the response generation. If you are the supporter, according to the conversation, please choose a proper strategy first from candidate strategies listed below:

{strategy_info}
When selecting a strategy, please take into account the three stages required to achieve emotional support: exploration, comfort, and action. Analyze which stages both the supporter and the help seeker are currently experiencing.
{knowledge}
'''
            strategy_info = ''
            for i in range(len(self.strategies)):
                # prompt += "{} {}".format(self.selections[i], self.strategies[i]) + " "  # + "({})\n".format(self.strategy_inf[self.strategies[i]])
                strategy_info += "{} {}".format(self.selections[i], self.strategies[i]) + " ({})\n".format(
                    self.strategy_inf[self.strategies[i]])

            for j in range(len(data_point['dialog']) - 1, -1, -1):
                if data_point['dialog'][j]['speaker'] == 'usr':
                    continue
                break

            data_point['dialog'] = data_point['dialog'][:j + 1]
            import random
            random.seed(random.randint(1, 1000))
            # random.shuffle(self.strategies)
            for i, dia in enumerate(data_point['dialog']):
                if i == len(data_point['dialog']) - 1:
                    # strategy = dia['strategy']
                    # strategy = dia['strategy']

                    # strategy = self.strategies[random.randint(0,len(self.strategies)-1)]
                    # strategy = 'Greeting'
                    # print(strategy,end=' ')
                    index = random.randint(0, len(self.selections) - 1)
                    # strategy = "{} {}".format(self.selections[index], self.strategies[index])
                    strategy = "{}".format(self.strategies[index])
                    output = dia['text']
                    break
                if dia['speaker'] == 'usr':
                    data_input += '({})Help seeker: '.format(i + 1)
                else:
                    data_input += '({})Supporter: '.format( i + 1)  # + 'Strategy chosen: ' + dia['annotation']['strategy']
                data_input += dia['text'].strip('\n') + '\n'

            # knowledge
            usr_last_utter = data_point['dialog'][-2]
            for i in range(len(data_point['dialog']) - 1, -1, -1):
                if data_point['dialog'][i]['speaker'] != 'sys' and 'ChatGPT_cause' in data_point['dialog'][i].keys() and data_point['dialog'][i]['ChatGPT_cause']:
                    usr_last_utter = data_point['dialog'][i]
                    break

            knowledge = \
'''
The subsequent event about the help seeker that happens or could happen following the last the utterance stated by the help seeker: {}

The possible emotional reaction of the help seeker in response to the last utterance stated by the help seeker is : {}

The underlying cause of the last utterance (the reason contributing to the utterance stated by the help seeker) is: {}
'''.format(usr_last_utter['ChatGPT_cause'].replace('\n', ''), usr_last_utter['ChatGPT_emo'].replace('\n', ''), usr_last_utter['ChatGPT_subs'].replace('\n', ''))

            if test:
                prompt += "The chosen strategy is: {strategy} " + "({})".format(self.strategy_inf[
                                                                                    strategy]) + "\n\nNow complete the conversation based on the picked strategy. The response of the conversation: "
                # prompt += "The chosen strategy is: {strategy} \n\nNow complete the conversation based on the picked strategy. The response of the conversation: "
                data.update({'input': data_input,'strategy_info':strategy_info, 'strategy': strategy,'knowledge': knowledge})
            else:
                prompt += "The chosen strategy is: {strategy} " + "({})".format(self.strategy_inf[
                                                                                    self.strategies[
                                                                                        index]]) + "\n\nNow complete the conversation based on the picked strategy. The response of the conversation: {output}"
                data.update({'input': data_input,'strategy_info':strategy_info, 'strategy': strategy, 'knowledge': knowledge,'output': output})
        else:
            '''There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress:

               {input}
               Up to now, the emotion supporter need to response properly to the seeker to make them feel better.
               Before responding, the supporter should follow some strategies to standardize the response generation. If you are the supporter, according to the conversation, please choose a proper strategy first from candidate strategies listed below:

               (a) Question
               (b) Restatement or Paraphrasing
               (c) Reflection of feelings
               (d) Self-disclosure
               (e) Affirmation and Reassurance
               (f) Providing Suggestions or Information
               (g) Greeting
               (h) Others

               The chosen strategy is:{output}'''
            prompt += \
'''There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress:

{input}
Up to now, the emotion supporter need to response properly to the seeker to make them feel better.
Before responding, the supporter should follow some strategies to standardize the response generation. If you are the supporter, according to the conversation, please choose a proper strategy first from candidate strategies listed below:

{strategy_info}
When selecting a strategy, please take into account the three stages required to achieve emotional support: exploration, comfort, and action. Analyze which stages both the supporter and the help seeker are currently experiencing.

{knowledge}

Now, please leverage the knowledge provided above to choose a properly strategy to make the help seeker feel better. '''

            import random
            random.seed(random.randint(1, 1000))
            random.shuffle(self.strategies)
            strategy_info = ''
            for i in range(len(self.strategies)):
                # prompt += "{} {}".format(self.selections[i], self.strategies[i]) + " "  # + "({})\n".format(self.strategy_inf[self.strategies[i]])
                strategy_info += "{} {}".format(self.selections[i], self.strategies[i]) + " ({})\n".format(
                    self.strategy_inf[self.strategies[i]])
                # input
            # prompt += "\n"
            for j in range(len(data_point['dialog']) - 1, -1, -1):
                if data_point['dialog'][j]['speaker'] == 'usr':
                    continue
                break
            data_point['dialog'] = data_point['dialog'][:j + 1]

            for i, dia in enumerate(data_point['dialog']):
                if i == len(data_point['dialog']) - 1:
                    try:
                        index = self.strategies.index(dia['strategy'])
                    except:
                        pdb.set_trace()
                    output = "{}".format(self.strategies[index])
                    break
                if dia['speaker'] == 'usr':
                    data_input += '({})Help seeker: '.format(i + 1)
                else:
                    data_input += '({})Supporter: '.format(
                        i + 1)  # + 'Strategy chosen: ' + dia['annotation']['strategy']
                data_input += dia['text'].strip('\n') + '\n'
            # prompt += 'When selecting a strategy, please take into account the three stages required to achieve emotional support: exploration, comfort, and action. Analyze which stages both the supporter and the help seeker are currently experiencing.  Let\' think step by step.\n'
            # knowledge
            usr_last_utter = data_point['dialog'][-2]
            for i in range(len(data_point['dialog']) - 1, -1, -1):
                try:
                    if data_point['dialog'][i]['speaker'] != 'sys' and 'ChatGPT_cause' in data_point['dialog'][i].keys() and data_point['dialog'][i]['ChatGPT_cause']:
                        usr_last_utter = data_point['dialog'][i]
                        break
                except:
                    pdb.set_trace()

            knowledge = \
'''Three subsequent event about the help seeker that happens or could happen following the last the utterance stated by the help seeker: 
{}

Three possible emotional reaction of the help seeker in response to the last utterance stated by the help seeker are: 
{}

Three underlying cause of the last utterance (the reason contributing to the utterance stated by the help seeker) are: 
{}'''.format(usr_last_utter['ChatGPT_cause'].replace('\n', ''), usr_last_utter['ChatGPT_emo'].replace('\n', ''),usr_last_utter['ChatGPT_subs'].replace('\n', ''))

            # response
            if test:
                prompt += "The chosen strategy is: "
                data.update({'input': data_input, 'strategy_info': strategy_info, 'knowledge': knowledge})
            else:
                prompt += "The chosen strategy is: {output}"
                data.update({'input': data_input, 'strategy_info': strategy_info, 'output': output, 'knowledge': knowledge})

        prompt = prompt.format_map(data)
        return prompt, output

# 915ED数据集生成四种CCI
class PrompterED_4CCI(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
        self.strategies = ["Question", "Restatement or Paraphrasing", "Reflection of feelings", "Self-disclosure",
                           "Affirmation and Reassurance", "Providing Suggestions or Information", "Greeting", "Others"]

    def extract_inf(self, inf_sentence):
        sentence_list = inf_sentence.split('\n')
        sentence_list = [sen for sen in sentence_list if sen != '' and ('1' in sen or '2' in sen or '3' in sen)]
        result = ' '.join(sentence_list[0].split(' ')[1:])
        pdb.set_trace()

        return result

    def generate_prompt(self, data_point, cci, test=False):
        prompt = ""
        output = ""
        data_input = ""
        data = {}

        if cci == 'ChatGPT_intent':
            prompt += \
'''You are an expert in the theory of empathy and conversational contextual reasoning. Given a dyadic dialogue clip between a listener and a speaker, the objective is to comprehend the dialogue in order to identify the emotional reaction of the speaker in their last utterance. Subsequently, you need to infer and analyze the listener's intent, taking into consideration the emotion reaction of the speaker.

I will provide an example, which is as follows:

(1)Speaker: I'm so sad because i've read an article about a newborn girl who died because her parents didn't believe in medication and doctors
(2)Listener: Ugh, those articles always get me too... :( What was wrong with her?
(3)Speaker: she was born premature at home, she had hard time breathing on her own but instead of taking her to the doctor parents were just praying

Emotion reaction of the speaker: Sad: The speaker feels a sense of compassion and sadness towards the unfortunate situation of the newborn girl and the failure of her parents to seek proper medical care for her.

What is the listener's intent to post the last utterance according to the emotion reaction of the speaker? Please infer and analyze the listener's intent conditioned on the speaker's emotion. 

Answer: The listener's intent is to connect with the speaker on an emotional level and offer comfort or a listening ear in response to the sad story the speaker shared.

Now, generate one concise and relevant inference (no more than 40 words) of the following conversation clip. The conversation clip is: 

{context}

Emotion reaction of the speaker: {emo}

What is the listener's intent to post the last utterance according to the emotion reaction of the speaker? Please infer and analyze the listener's intent conditioned on speaker's emotion reaction.

'''
        elif cci == 'ChatGPT_cause':
            prompt += \
'''You are an expert in the theory of empathy and conversational contextual reasoning. Given a dyadic dialogue clip between a listener and a speaker, the objective is to comprehend the dialogue and make inferences to identify the underlying cause of the latest utterance stated by the listener (the reason contributing to the utterance stated by the listener).

I will provide an example of a conversation clip and the explanation of causes, which is as follows:

(1)Speaker: Job interviews always make me sweat bullets, makes me uncomfortable in general to be looked at under a microscope like that.
(2)Listener: Don't be nervous. Just be prepared.
(3)Speaker: I feel like getting prepared and then having a curve ball thrown at you throws you off.

What would be the cause of listener to post the next utterance? Please make inference based on the utterances before the last utterance of the conversation. Please generate the answer like this: Answer: The cause of the listener's next utterance is to provide reassurance and encouragement to the speaker, emphasizing the importance of staying calm during job interviews to cope with unexpected challenges and alleviate the speaker's anxiety.

Now, generate one concise and relevant inference (no more than 40 words) of the cause of the last utterance. The conversation clip is: 
{context}

What would be the cause of listener to post the next utterance?

'''
        elif cci == 'ChatGPT_emo':
            prompt += \
'''You are an expert in the theory of empathy and conversational contextual reasoning. Given a dyadic dialogue clip between a listener and a speaker, the objective is to comprehend the dialogue in order to identify the emotional reaction of the speaker in their last utterance.

I will provide an example, which is as follows:

(1)Speaker: Hi, this year, I was the first over 300 students at my engineering school
(2)Listener: Sounds great! So what's your major?
(3)Speaker: It is computer science. I am very happy of this achievement and my family is very proud.

What is the emotional reaction of the speaker in their last utterance? Please generate the answer like this: Answer: Happy: The speaker feel pleased and proud of their achievement and appreciate the listener's recognition of their hard work in their computer science major.

Now, generate one concise and relevant inference (no more than 40 words) about the emotional reaction of the speaker of the last utterance. The conversation clip is: 

{context}

What is the emotional reaction of the speaker in their last utterance?

'''
        elif cci == 'ChatGPT_subs':
            prompt += \
'''You are an expert in the theory of empathy and conversational contextual reasoning. Given a dyadic dialogue clip between a listener and a speaker, the objective is to comprehend the dialogue context and make inferences about potential subsequent events involving the listener that may occur after the speaker's last utterance.

I will provide an example, which is as follows:
(1)Speaker: Job interviews always make me sweat bullets, makes me uncomfortable in general to be looked at under a microscope like that.
(2)Listener: Don't be nervous. Just be prepared.
(3)Speaker: I feel like getting prepared and then having a curve ball thrown at you throws you off.

What is the subsequent event potential subsequent events involving the listener that may occur after the speaker's last utterance? Please generate the answer like this: Answer: The listener may offer specific strategies or advice on how to stay calm during job interviews, such as mindfulness techniques or sharing personal experiences to further reassure and help the speaker manage their anxiety.

Now, generate one concise and relevant inference (no more than 40 words) about subsequent events involving the listener that may occur after the speaker's last utterance. The conversation clip is: 

{context}

What is the subsequent event potential subsequent events involving the listener that may occur after the speaker's last utterance?

'''
#         if cci == 'ChatGPT_intent':
#             prompt += \
# '''You are an expert in the theory of empathy and conversational contextual reasoning. Given a dyadic dialogue clip between a listener and a speaker, the objective is to comprehend the dialogue in order to identify the emotional reaction of the speaker in their last utterance. Subsequently, you need to infer and analyze the listener's intent, taking into consideration the emotion reaction of the speaker.
#
# I will provide an example, which is as follows:
#
# (1)Speaker: I'm so sad because i've read an article about a newborn girl who died because her parents didn't believe in medication and doctors
# (2)Listener: Ugh, those articles always get me too... :( What was wrong with her?
# (3)Speaker: she was born premature at home, she had hard time breathing on her own but instead of taking her to the doctor parents were just praying
# (4)Listener: Jeez! Its so unfortunate... very sad really.
#
# Emotion reaction of the speaker: Sad: The speaker feels a sense of compassion and sadness towards the unfortunate situation of the newborn girl and the failure of her parents to seek proper medical care for her.
#
# What is the listener's intent to post the last utterance according to the emotion reaction of the speaker? Please infer and analyze the listener's intent conditioned on the speaker's emotion.
#
# Answer: The listener's intent is to connect with the speaker on an emotional level and offer comfort or a listening ear in response to the sad story the speaker shared.
#
# Now, generate one concise and relevant inference (no more than 40 words) of the following conversation clip. The conversation clip is:
#
# {context}
#
# Emotion reaction of the speaker: {emo}
#
# What is the listener's intent to post the last utterance according to the emotion reaction of the speaker? Please infer and analyze the listener's intent conditioned on speaker's emotion reaction.
#
# '''
#         elif cci == 'ChatGPT_cause':
#             prompt += \
# '''You are an expert in the theory of empathy and conversational contextual reasoning. Given a dyadic dialogue clip between a listener and a speaker, the objective is to comprehend the dialogue and make inferences to identify the underlying cause of the latest utterance stated by the listener (the reason contributing to the utterance stated by the listener).
#
# I will provide an example of a conversation clip and the explanation of causes, which is as follows:
#
# (1)Speaker: Job interviews always make me sweat bullets, makes me uncomfortable in general to be looked at under a microscope like that.
# (2)Listener: Don't be nervous. Just be prepared.
# (3)Speaker: I feel like getting prepared and then having a curve ball thrown at you throws you off.
# (4)Listener: Yes but if you stay calm it will be ok.
#
# What is the cause of speaker to post the last utterance? Please make inference based on the utterances before the last utterance of the conversation. Please generate the answer like this: Answer: The cause of the listener's last utterance is to provide reassurance and encouragement to the speaker, emphasizing the importance of staying calm during job interviews to cope with unexpected challenges and alleviate the speaker's anxiety.
#
# Now, generate one concise and relevant inference (no more than 40 words) of the cause of the last utterance. The conversation clip is:
# {context}
#
# What is the cause of speaker to post the last utterance?
#
# '''
#         elif cci == 'ChatGPT_emo':
#             prompt += \
# '''You are an expert in the theory of empathy and conversational contextual reasoning. Given a dyadic dialogue clip between a listener and a speaker, the objective is to comprehend the dialogue in order to identify the emotional reaction of the speaker in their last utterance.
#
# I will provide an example, which is as follows:
#
# (1)Speaker: Hi, this year, I was the first over 300 students at my engineering school
# (2)Listener: Sounds great! So what's your major?
# (3)Speaker: It is computer science. I am very happy of this achievement and my family is very proud.
# (4)Listener: Well pleased. You should be having brains,man!That's a tough course,i hear.
#
# What is the emotional reaction of the speaker in their last utterance? Please generate the answer like this: Answer: Happy: The speaker feel pleased and proud of their achievement and appreciate the listener's recognition of their hard work in their computer science major.
#
# Now, generate one concise and relevant inference (no more than 40 words) about the emotional reaction of the speaker of the last utterance. The conversation clip is:
#
# {context}
#
# What is the emotional reaction of the speaker in their last utterance?
#
# '''
#         elif cci == 'ChatGPT_subs':
#             prompt += \
# '''You are an expert in the theory of empathy and conversational contextual reasoning. Given a dyadic dialogue clip between a listener and a speaker, the objective is to comprehend the dialogue context and make inferences about potential subsequent events involving the listener that may occur after the speaker's last utterance.
#
# I will provide an example, which is as follows:
# (1)Speaker: Job interviews always make me sweat bullets, makes me uncomfortable in general to be looked at under a microscope like that.
# (2)Listener: Don't be nervous. Just be prepared.
# (3)Speaker: I feel like getting prepared and then having a curve ball thrown at you throws you off.
# (4)Listener: Yes but if you stay calm it will be ok.
#
# What is the subsequent event potential subsequent events involving the listener that may occur after the speaker's last utterance? Please generate the answer like this: Answer: The listener may offer specific strategies or advice on how to stay calm during job interviews, such as mindfulness techniques or sharing personal experiences to further reassure and help the speaker manage their anxiety.
#
# Now, generate one concise and relevant inference (no more than 40 words) about subsequent events involving the listener that may occur after the speaker's last utterance. The conversation clip is:
#
# {context}
#
# What is the subsequent event potential subsequent events involving the listener that may occur after the speaker's last utterance?
#
# '''
        output = data_point[cci]

        if "Answer:" == output[:len("Answer:")]:
            output = output[len("Answer:"):].strip()
        elif "Inference:" == output[:len("Inference:")]:
            output = output[len("Inference:"):].strip()

        if "The cause of the listener's last utterance" in output:
            output = output.replace("The cause of the listener's last utterance", "The cause of the listener's next utterance")

        emo = data_point['ChatGPT_emo']
        data_input = ''
        for i, ut in enumerate(data_point['context']):
            data_input += '({})'.format(i + 1)
            if i % 2 == 0:
                data_input += 'Speaker: ' + ut + '\n'
            else:
                data_input += 'Listener: ' + ut + '\n'

        # data_input += '({})'.format(len(data_point['context']) + 1) + 'Listener: ' + data_point['target']

        if test:
            prompt += "Answer: "
            if cci == 'ChatGPT_intent':
                data.update({'context': data_input, "emo": emo})
            else:
                data.update({'context': data_input})
        else:
            prompt += "Answer: {output}"
            if cci == 'ChatGPT_intent':
                data.update({'context': data_input, 'output': output,"emo":emo})
            else:
                data.update({'context': data_input,'output': output})

        prompt = prompt.format_map(data)
        return prompt, output

class Prompter4CCI(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
        # self.selections = ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)"]
        # self.selections = ["1", "2", "3", "4", "5", "6", "7", "8"]
        self.strategies = ["Question", "Restatement or Paraphrasing", "Reflection of feelings", "Self-disclosure",
                           "Affirmation and Reassurance", "Providing Suggestions or Information", "Greeting", "Others"]
        self.strategy_inf = {
            "Question": "This emotion support strategy involves inquiring about details concerning the problem to aid the individual seeking help in expressing their specific challenges. Open-ended questions are highly effective in this regard, while closed questions can be employed to obtain more precise information.",
            "Restatement or Paraphrasing": "Restating or paraphrasing refers to the act of succinctly rephrasing the help-seeker's statements, which can assist them in gaining a clearer perspective on their situation.",
            "Reflection of feelings": "The technique of reflecting feelings involves effectively expressing and describing the emotions experienced by the individual seeking help.",
            "Self-disclosure": "Self-disclosure entails sharing relevant personal experiences or emotions that resonate with the help-seeker, thus demonstrating empathy.",
            "Affirmation and Reassurance": "Affirmation and reassurance within the emotion support strategy involve acknowledging and validating the help-seeker's strengths, motivation, and abilities, while also offering reassurance and encouragement.",
            "Providing Suggestions or Information": "The aspect of providing suggestions or information within the emotion support strategy involves offering recommendations on potential changes, while being cautious not to dictate or prescribe specific actions. It also encompasses providing helpful information to the help-seeker, such as data, facts, opinions, resources, or responding to their inquiries.",
            "Greeting": "Greeting is about exchange pleasantries.",
            "Others": "Use other support strategies that do not fall into the above categories.",
        }

    def extract_inf(self, inf_sentence):
        sentence_list = inf_sentence.split('\n')
        sentence_list = [sen for sen in sentence_list if sen != '' and ('1' in sen or '2' in sen or '3' in sen)]
        result = ' '.join(sentence_list[0].split(' ')[1:])
        pdb.set_trace()

        return result

    def generate_prompt(self, data_point, cci, test=False):
        prompt = ""
        output = ""
        data_input = ""
        data = {}

        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break
        data_point['dialog'] = data_point['dialog'][:j + 1]
        if cci == 'ChatGPT_intent':
            prompt += \
'''Given a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress, the objective is to comprehend the dialogue in order to identify the emotional reaction of the help seeker in their last utterance. Subsequently, you need to infer and analyze the supporter's intent, taking into consideration the emotion reaction of the help seeker.

I will provide an example, which is as follows:

(1)Help seeker: I am laid off .
(2)Supporter: Oh I am really sorry to hear that , Did you have the same job for a long time ? That sounds very difficult to deal with .
(3)Help seeker: I was attending a customer . He was having a grievance which I sorted out . But the management did not like that . For the past 10 years I have been with this job .
(4)Supporter: 10 years is a very long time and I would think you have probably learned a lot working at the same place for that long . You ' re a dedicated employee .
(5)Help seeker: I was okay with the previous manager . But recently a new young chap joined the duty . He was not experienced . He only is the reason for so .
(6)Supporter: That is really unfair and hard to deal with are you close to any family ?
(7)Help seeker: I am not in any close to any family related to job .
(8)Supporter: Do you have any close friends to talk to about any new job prospects ?
(9)Help seeker: I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .

Emotion reaction of the help seeker: Optimism: The help seeker expresses optimism about their job prospects, feeling hopeful and confident in their skills and qualifications to secure a better job.

What is the supporter's intent to post the last utterance according to the emotion reaction of the help seeker? Please infer and analyze the supporter's intent conditioned on helper seeker's emotion. 

Answer: The supporter's intent is to reinforce the help seeker's optimism and positive outlook. By suggesting joining a new group at a church or something similar, the supporter aims to provide additional emotional support and offer opportunities for the help seeker to connect with others who may provide guidance and new job prospects. 

Now, generate one concise and relevant inference (no more than 40 words) of the following conversation clip. The conversation clip is: 

{context}

Emotion reaction of the help seeker: {emo}

What is the supporter's intent to post the last utterance according to the emotion reaction of the help seeker? Please infer and analyze the supporter's intent conditioned on helper seeker's emotion reaction.

'''
        elif cci == 'ChatGPT_cause':
            prompt += \
'''Given a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress, the objective is to comprehend the dialogue and make inferences to identify the underlying cause of the latest utterance stated by the supporter (the reason contributing to the utterance stated by the supporter).

I will provide an example of a conversation clip and the explanation of causes, which is as follows:

(1)Help seeker: I am laid off .
(2)Supporter: Oh I am really sorry to hear that , Did you have the same job for a long time ? That sounds very difficult to deal with .
(3)Help seeker: I was attending a customer . He was having a grievance which I sorted out . But the management did not like that . For the past 10 years I have been with this job .
(4)Supporter: 10 years is a very long time and I would think you have probably learned a lot working at the same place for that long . You ' re a dedicated employee .
(5)Help seeker: I was okay with the previous manager . But recently a new young chap joined the duty . He was not experienced . He only is the reason for so .
(6)Supporter: That is really unfair and hard to deal with are you close to any family ?
(7)Help seeker: I am not in any close to any family related to job .
(8)Supporter: Do you have any close friends to talk to about any new job prospects ?
(9)Help seeker: I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .

What is the cause of supporter to post the last utterance? Please make inference based on the utterances before the last utterance of the conversation. Please generate the answer like this: Answer: The supporter recognizes that the help seeker is facing unfair treatment at work due to the new inexperienced manager, which is causing emotional distress. The suggestion of joining a new group at a church might be a way to provide a supportive environment outside of the workplace.

Now, generate one concise and relevant inference (no more than 40 words) of the cause of the last utterance. The conversation clip is: 

{context}
'''
        elif cci == 'ChatGPT_emo':
            prompt += \
'''Given a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress, the objective is to comprehend the dialogue in order to identify the emotional reaction of the help seeker in their last utterance.

I will provide an example, which is as follows:

(1)Help seeker: I am laid off .
(2)Supporter: Oh I am really sorry to hear that , Did you have the same job for a long time ? That sounds very difficult to deal with .
(3)Help seeker: I was attending a customer . He was having a grievance which I sorted out . But the management did not like that . For the past 10 years I have been with this job .
(4)Supporter: 10 years is a very long time and I would think you have probably learned a lot working at the same place for that long . You ' re a dedicated employee .
(5)Help seeker: I was okay with the previous manager . But recently a new young chap joined the duty . He was not experienced . He only is the reason for so .
(6)Supporter: That is really unfair and hard to deal with are you close to any family ?
(7)Help seeker: I am not in any close to any family related to job .
(8)Supporter: Do you have any close friends to talk to about any new job prospects ?
(9)Help seeker: I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .

What is the emotional reaction of the help seeker in their last utterance? Please generate the answer like this: Answer: Optimism: The help seeker expresses optimism about their job prospects, feeling hopeful and confident in their skills and qualifications to secure a better job.

Now, generate one concise and relevant inference (no more than 40 words) about the emotional reaction of the help seeker of the last utterance. The conversation clip is:  

{context}
'''
        elif cci == 'ChatGPT_subs':
            prompt += \
'''Given a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress, the objective is to comprehend the dialogue context and make inferences about potential subsequent events involving the supporter that may occur after the help seeker's last utterance.

I will provide an example, which is as follows:

(1)Help seeker: I am laid off .
(2)Supporter:: Oh I am really sorry to hear that , Did you have the same job for a long time ? That sounds very difficult to deal with .
(3)Help seeker: I was attending a customer . He was having a grievance which I sorted out . But the management did not like that . For the past 10 years I have been with this job .
(4)Supporter: 10 years is a very long time and I would think you have probably learned a lot working at the same place for that long . You ' re a dedicated employee .
(5)Help seeker: I was okay with the previous manager . But recently a new young chap joined the duty . He was not experienced . He only is the reason for so .
(6)Supporter: That is really unfair and hard to deal with are you close to any family ?
(7)Help seeker: I am not in any close to any family related to job .
(8)Supporter: Do you have any close friends to talk to about any new job prospects ?
(9)Help seeker:I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .
(10)Supporter: Do you have any close friends to talk to about any new job prospects ?
(11)Help seeker: I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .
(12)Supporter: That is a positive outlook and is good to hear that you know you have skills to offer . Would you consider joining a new group at a church or something like that ?
(13)Help seeker: I am so worried of the management taken action on me relying a new inexperienced manager . .
(14)Supporter: I am sorry that you ' re feeling stress . Have you ever used writing as a tool to relax ?
(15)Help seeker: My colleagues are also in contact with me . They are also having similar inconvenience as to how to perform ? I have some other relaxation like listening to music , gardening etc . ,

What is the subsequent event potential subsequent events involving the supporter that may occur after the help seeker's last utterance? Please generate the answer like this: Answer: The supporter could recommend specific relaxation techniques involving music or gardening to further enhance the help seeker's coping mechanisms.

Now, generate one concise and relevant inference (no more than 40 words) about subsequent events involving the supporter that may occur after the help seeker's last utterance. The conversation clip is:

{context}
'''
#         if cci == 'ChatGPT_intent':
#             prompt += \
# '''Given a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress, the objective is to comprehend the dialogue in order to identify the emotional reaction of the help seeker in their last utterance. Subsequently, you need to infer and analyze the supporter's intent, taking into consideration the emotion reaction of the help seeker.
#
# I will provide an example, which is as follows:
#
# (1)Help seeker: I am laid off .
# (2)Supporter: Oh I am really sorry to hear that , Did you have the same job for a long time ? That sounds very difficult to deal with .
# (3)Help seeker: I was attending a customer . He was having a grievance which I sorted out . But the management did not like that . For the past 10 years I have been with this job .
# (4)Supporter: 10 years is a very long time and I would think you have probably learned a lot working at the same place for that long . You ' re a dedicated employee .
# (5)Help seeker: I was okay with the previous manager . But recently a new young chap joined the duty . He was not experienced . He only is the reason for so .
# (6)Supporter: That is really unfair and hard to deal with are you close to any family ?
# (7)Help seeker: I am not in any close to any family related to job .
# (8)Supporter: Do you have any close friends to talk to about any new job prospects ?
# (9)Help seeker: I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .
# (10)Supporter: That is a positive outlook and is good to hear that you know you have skills to offer . Would you consider joining a new group at a church or something like that?
#
# Emotion reaction of the help seeker: Optimism: The help seeker expresses optimism about their job prospects, feeling hopeful and confident in their skills and qualifications to secure a better job.
#
# What is the supporter's intent to post the last utterance according to the emotion reaction of the help seeker? Please infer and analyze the supporter's intent conditioned on helper seeker's emotion.
#
# Answer: The supporter's intent is to reinforce the help seeker's optimism and positive outlook. By suggesting joining a new group at a church or something similar, the supporter aims to provide additional emotional support and offer opportunities for the help seeker to connect with others who may provide guidance and new job prospects.
#
# Now, generate one concise and relevant inference (no more than 40 words) of the following conversation clip. The conversation clip is:
#
# {context}
#
# Emotion reaction of the help seeker: {emo}
#
# What is the supporter's intent to post the last utterance according to the emotion reaction of the help seeker? Please infer and analyze the supporter's intent conditioned on helper seeker's emotion reaction.
#
# '''
#         elif cci == 'ChatGPT_cause':
#             prompt += \
# '''Given a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress, the objective is to comprehend the dialogue and make inferences to identify the underlying cause of the latest utterance stated by the supporter (the reason contributing to the utterance stated by the supporter).
#
# I will provide an example of a conversation clip and the explanation of causes, which is as follows:
#
# (1)Help seeker: I am laid off .
# (2)Supporter: Oh I am really sorry to hear that , Did you have the same job for a long time ? That sounds very difficult to deal with .
# (3)Help seeker: I was attending a customer . He was having a grievance which I sorted out . But the management did not like that . For the past 10 years I have been with this job .
# (4)Supporter: 10 years is a very long time and I would think you have probably learned a lot working at the same place for that long . You ' re a dedicated employee .
# (5)Help seeker: I was okay with the previous manager . But recently a new young chap joined the duty . He was not experienced . He only is the reason for so .
# (6)Supporter: That is really unfair and hard to deal with are you close to any family ?
# (7)Help seeker: I am not in any close to any family related to job .
# (8)Supporter: Do you have any close friends to talk to about any new job prospects ?
# (9)Help seeker: I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .
# (10)Supporter: That is a positive outlook and is good to hear that you know you have skills to offer . Would you consider joining a new group at a church or something like that ?
#
# What is the cause of supporter to post the last utterance? Please make inference based on the utterances before the last utterance of the conversation. Please generate the answer like this: Answer: The supporter recognizes that the help seeker is facing unfair treatment at work due to the new inexperienced manager, which is causing emotional distress. The suggestion of joining a new group at a church might be a way to provide a supportive environment outside of the workplace.
#
# Now, generate one concise and relevant inference (no more than 40 words) of the cause of the last utterance. The conversation clip is:
#
# {context}
# '''
#         elif cci == 'ChatGPT_emo':
#             prompt += \
# '''Given a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress, the objective is to comprehend the dialogue in order to identify the emotional reaction of the help seeker in their last utterance.
#
# I will provide an example, which is as follows:
#
# (1)Help seeker: I am laid off .
# (2)Supporter: Oh I am really sorry to hear that , Did you have the same job for a long time ? That sounds very difficult to deal with .
# (3)Help seeker: I was attending a customer . He was having a grievance which I sorted out . But the management did not like that . For the past 10 years I have been with this job .
# (4)Supporter: 10 years is a very long time and I would think you have probably learned a lot working at the same place for that long . You ' re a dedicated employee .
# (5)Help seeker: I was okay with the previous manager . But recently a new young chap joined the duty . He was not experienced . He only is the reason for so .
# (6)Supporter: That is really unfair and hard to deal with are you close to any family ?
# (7)Help seeker: I am not in any close to any family related to job .
# (8)Supporter: Do you have any close friends to talk to about any new job prospects ?
# (9)Help seeker: I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .
# (10)Supporter: That is a positive outlook and is good to hear that you know you have skills to offer . Would you consider joining a new group at a church or something like that ?
#
# What is the emotional reaction of the help seeker in their last utterance? Please generate the answer like this: Answer: Optimism: The help seeker expresses optimism about their job prospects, feeling hopeful and confident in their skills and qualifications to secure a better job.
#
# Now, generate one concise and relevant inference (no more than 40 words) about the emotional reaction of the help seeker of the last utterance. The conversation clip is:
#
# {context}
# '''
#         elif cci == 'ChatGPT_subs':
#             prompt += \
# '''Given a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress, the objective is to comprehend the dialogue context and make inferences about potential subsequent events involving the supporter that may occur after the help seeker's last utterance.
#
# I will provide an example, which is as follows:
#
# (1)Help seeker: I am laid off .
# (2)Supporter:: Oh I am really sorry to hear that , Did you have the same job for a long time ? That sounds very difficult to deal with .
# (3)Help seeker: I was attending a customer . He was having a grievance which I sorted out . But the management did not like that . For the past 10 years I have been with this job .
# (4)Supporter: 10 years is a very long time and I would think you have probably learned a lot working at the same place for that long . You ' re a dedicated employee .
# (5)Help seeker: I was okay with the previous manager . But recently a new young chap joined the duty . He was not experienced . He only is the reason for so .
# (6)Supporter: That is really unfair and hard to deal with are you close to any family ?
# (7)Help seeker: I am not in any close to any family related to job .
# (8)Supporter: Do you have any close friends to talk to about any new job prospects ?
# (9)Help seeker:I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .
# (10)Supporter: Do you have any close friends to talk to about any new job prospects ?
# (11)Help seeker: I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .
# (12)Supporter: That is a positive outlook and is good to hear that you know you have skills to offer . Would you consider joining a new group at a church or something like that ?
# (13)Help seeker: I am so worried of the management taken action on me relying a new inexperienced manager . .
# (14)Supporter: I am sorry that you ' re feeling stress . Have you ever used writing as a tool to relax ?
# (15)Help seeker: My colleagues are also in contact with me . They are also having similar inconvenience as to how to perform ? I have some other relaxation like listening to music , gardening etc . ,
# (16)Supporter: Oh , wow then you do have a lot of contacts and some support or at least understanding . Music is a great way to relax and that id very positive in your life and current situation .
#
# What is the subsequent event potential subsequent events involving the supporter that may occur after the help seeker's last utterance? Please generate the answer like this: Answer: The supporter could recommend specific relaxation techniques involving music or gardening to further enhance the help seeker's coping mechanisms.
#
# Now, generate one concise and relevant inference (no more than 40 words) about subsequent events involving the supporter that may occur after the help seeker's last utterance. The conversation clip is:
#
# {context}
# '''
        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break

        data_point['dialog'] = data_point['dialog'][:j + 1]
        output = data_point['dialog'][-2][cci]

        if "Answer:" == output[:len("Answer:")]:
            output = output[len("Answer:"):].strip()
        elif "Inference:" == output[:len("Inference:")]:
            output = output[len("Inference:"):].strip()

        emo = data_point['dialog'][-2]["ChatGPT_emo"]
        for i, dia in enumerate(data_point['dialog']):
            if i == len(data_point['dialog']) - 1:
                break
            if dia['speaker'] == 'usr':
                data_input += '({})Help seeker: '.format(i + 1)
            else:
                data_input += '({})Supporter: '.format( i + 1)  # + 'Strategy chosen: ' + dia['annotation']['strategy']
            data_input += dia['text'].strip('\n') + '\n'


        if test:
            prompt += "Answer: "
            if cci == 'ChatGPT_intent':
                data.update({'context': data_input, "emo":emo})
            else:
                data.update({'context': data_input})
        else:
            prompt += "Answer: {output}"
            if cci == 'ChatGPT_intent':
                data.update({'context': data_input, 'output': output,"emo":emo})
            else:
                data.update({'context': data_input,'output': output})


        prompt = prompt.format_map(data)
        return prompt, output

#821用四种cci enhance ESC回复生成
class Prompter_template_4CCIESC(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    def get_prompt(self,message: str, chat_history: list[tuple[str, str]],system_prompt: str) -> str:

        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        for user_input, response in chat_history:
            texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
        if not message == "":
            texts.append(f'{message.strip()} [/INST]')
        return ''.join(texts)

    def generate_prompt(self, data_point, test=False, comet=False):
        data = {}

        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break
        data_point['dialog'] = data_point['dialog'][:j + 1]

        # knowledge
        if not comet:
            usr_last_utter = data_point['dialog'][-2]

            cause = usr_last_utter['ChatGPT_cause']
            emo = usr_last_utter['ChatGPT_emo']
            subs = usr_last_utter['ChatGPT_subs']
            intent = usr_last_utter['ChatGPT_intent']
            if "Answer:" == cause[:len("Answer:")]:
                cause = cause[len("Answer:"):].strip()
            elif "Inference:" == cause[:len("Inference:")]:
                cause = cause[len("Inference:"):].strip()

            if "Answer:" == emo[:len("Answer:")]:
                emo = emo[len("Answer:"):].strip()
            elif "Inference:" == emo[:len("Inference:")]:
                emo = emo[len("Inference:"):].strip()

            if subs[:len("Answer:")] == "Answer:":
                subs = subs[len("Answer:"):].strip()
            elif "Inference:" == subs[:len("Inference:")]:
                subs = subs[len("Inference:"):].strip()

            if "Answer:" == intent[:len("Answer:")]:
                intent = intent[len("Answer:"):].strip()
            elif "Inference:" == intent[:len("Inference:")]:
                intent = intent[len("Inference:"):].strip()

            knowledge = \
'''
The underlying cause of the help seeker's last utterance (the reason contributing to the utterance stated by the help seeker) is: {}

The subsequent event about the supporter that happens or could happen following the last the utterance stated by the help seeker :{}

The possible emotional reaction of the help seeker in response to the last utterance stated by the help seeker is : {}

The supporter's intent to post the last utterance according to the emotion reaction of the help seeker is : {}
'''.format(cause, subs, emo, intent)
# '''
# The underlying cause of the last utterance (the reason contributing to the utterance stated by the user) is: {}
#
# The subsequent event about the help seeker that happens or could happen following the last the utterance stated by the user :{}
#
# The possible emotional reaction of the help seeker in response to the last utterance stated by the user is : {}
#
# The supporter's intent to post the last utterance according to the emotion reaction of the user is : {}
# '''.format(cause, subs, emo, intent)

# No cause
# '''
# The subsequent event about the help seeker that happens or could happen following the last the utterance stated by the user :{}
#
# The possible emotional reaction of the help seeker in response to the last utterance stated by the user is : {}
#
# The supporter's intent to post the last utterance according to the emotion reaction of the user is : {}
# '''.format(subs, emo, intent)

# No subs
# '''
# The underlying cause of the last utterance (the reason contributing to the utterance stated by the user) is: {}
#
# The possible emotional reaction of the help seeker in response to the last utterance stated by the user is : {}
#
# The supporter's intent to post the last utterance according to the emotion reaction of the user is : {}
# '''.format(cause, emo, intent)

# No emo
# '''
# The underlying cause of the last utterance (the reason contributing to the utterance stated by the user) is: {}
#
# The subsequent event about the help seeker that happens or could happen following the last the utterance stated by the user :{}
#
# The supporter's intent to post the last utterance according to the emotion reaction of the user is : {}
# # '''.format(cause, subs, intent)


        else:
            relations = ["oIntent", "oNeed", "oWant", "oEffect", "oReact"]
            last_utt_intent = ' or '.join([' '.join(i) for i in data_point["dialog"][-2]['COMET']['oIntent'][:3] if i != ['none']]).replace(".", "")
            last_utt_need = ' or '.join([' '.join(i) for i in data_point["dialog"][-2]['COMET']['oNeed'][:3] if i != ['none']]).replace(".", "")
            last_utt_want = ' or '.join([' '.join(i) for i in data_point["dialog"][-2]['COMET']['oWant'][:3] if i != ['none']]).replace(".", "")
            last_utt_effect = ' or '.join([' '.join(i) for i in data_point["dialog"][-2]['COMET']['oEffect'][:3] if i != ['none']]).replace(".", "")
            last_utt_react = ' or '.join([' '.join(i) for i in data_point["dialog"][-2]['COMET']['oReact'][:3] if i != ['none']]).replace(".", "")
            knowledge = 'After posting the last utterance, the supporter intent to {}, need to {} and want to {}, the supporter may feel {}, additionally the supporter would {}.'.format(last_utt_intent, last_utt_need, last_utt_want,
                                                                                                                                                                                   last_utt_react, last_utt_effect)

        chat_history = []
        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break
        data_point['dialog'] = data_point['dialog'][:j + 1]

        char_flag = 0  # 0代表usr 1代表sys
        usr, sys = "", ""
        for i, dia in enumerate(data_point['dialog']):
            if i == len(data_point['dialog']) - 1:
                output = dia['text']
            if dia['speaker'] == 'usr':
                if char_flag == 1:
                    chat_history.append((usr, sys))
                    usr, sys = "", ""
                usr += dia['text'].strip()
                char_flag = 0
            else:
                sys += dia['text'].strip()
                char_flag = 1
        # system_prompt = "There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress.\n"
        system_prompt = "You are an expert in the theory of emotional support and conversational contextual reasoning. " \
                        "You are well aware that emotional support follows a three-stage process: exploration, providing comfort, and taking action. You possess the expertise to skillfully choose the appropriate strategy to gradually alleviate the negative emotions of those seeking help. " \
                        "There is a dyadic dialogue clip between an emotional supporter (assistant) and a help seeker (user) who seeks for help in relieving emotional distress.\n" \
                        "Please generate a response that incorporates relevant common-sense knowledge: " + knowledge
        if test:
            message = usr
            prompt = self.get_prompt(message,chat_history,system_prompt)
        else:
            chat_history.append((usr, sys))
            prompt = self.get_prompt("", chat_history, system_prompt)


        prompt = prompt.format_map(data)
        return prompt, output

class Prompter_template_4CCIED(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    def get_prompt(self,message: str, chat_history: list[tuple[str, str]],system_prompt: str) -> str:

        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        for user_input, response in chat_history:
            texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
        if not message == "":
            texts.append(f'{message.strip()} [/INST] ')
        return ''.join(texts)

    def generate_prompt(self, data_point, test=False, comet=False):

        # knowledge
        if not comet:
            cause = data_point['ChatGPT_cause']
            emo = data_point['ChatGPT_emo']
            subs = data_point['ChatGPT_subs']
            intent = data_point['ChatGPT_intent']
            if "Answer:" == cause[:len("Answer:")]:
                cause = cause[len("Answer:"):].strip()
            elif "Inference:" == cause[:len("Inference:")]:
                cause = cause[len("Inference:"):].strip()
            if "The cause of the listener's last utterance" in cause:
                cause = cause.replace("The cause of the listener's last utterance", "The cause of the listener's next utterance")
            if "Answer:" == emo[:len("Answer:")]:
                emo = emo[len("Answer:"):].strip()
            elif "Inference:" == emo[:len("Inference:")]:
                emo = emo[len("Inference:"):].strip()

            if subs[:len("Answer:")] == "Answer:":
                subs = subs[len("Answer:"):].strip()
            elif "Inference:" == subs[:len("Inference:")]:
                subs = subs[len("Inference:"):].strip()
            # if 'supporter' in subs:
            #     subs = subs.replace('supporter','listener')

            if "Answer:" == intent[:len("Answer:")]:
                intent = intent[len("Answer:"):].strip()
            elif "Inference:" == intent[:len("Inference:")]:
                intent = intent[len("Inference:"):].strip()

            knowledge = \
'''
The underlying cause of the listener's next utterance (the reason contributing to response) is: {}

The subsequent event about the listener that happens or could happen following the last the utterance stated by the listener: {}

The possible emotional reaction of the speaker in response to the last utterance stated by the speaker is: {}

The listener's intent to post the last utterance according to the emotion reaction of the speaker is: {}
'''.format(cause, subs, emo, intent)

# '''
# The subsequent event about the listener that happens or could happen following the last the utterance stated by the assistant: {}
# '''.format(subs)

# '''
# The listener's intent to post the last utterance according to the emotion reaction of the user is: {}
# '''.format(intent)

# '''
# The possible emotional reaction of the speaker in response to the last utterance stated by the user is: {}
# '''.format(emo)

# '''
# The subsequent event about the listener that happens or could happen following the last the utterance stated by the assistant: {}
# '''.format(subs)

# '''
# The underlying cause of the speaker's last utterance (the reason contributing to the utterance stated by the user) is: {}
# '''.format(cause)

# '''
# {}
#
# {}
#
# {}
#
# {}
# '''.format(cause, subs, emo, intent)

# '''
# The underlying cause of the listener's next utterance (the reason contributing to response) is: {}
#
# The subsequent event about the listener that happens or could happen following the last the utterance stated by the listener: {}
#
# The possible emotional reaction of the speaker in response to the last utterance stated by the speaker is: {}
#
# The listener's intent to post the last utterance according to the emotion reaction of the speaker is: {}
# '''.format(cause, subs, emo, intent)

# No cause
# '''
# The subsequent event about the listener that happens or could happen following the last the utterance stated by the assistant: {}
#
# The possible emotional reaction of the speaker in response to the last utterance stated by the user is: {}
#
# The listener's intent to post the last utterance according to the emotion reaction of the user is: {}
# '''.format(subs, emo, intent)

# No subs
# '''
# The underlying cause of the speaker's last utterance (the reason contributing to the utterance stated by the user) is: {}
#
# The possible emotional reaction of the speaker in response to the last utterance stated by the user is: {}
#
# The listener's intent to post the last utterance according to the emotion reaction of the user is: {}
# '''.format(cause, emo, intent)

# No emo
# '''
# The underlying cause of the speaker's last utterance (the reason contributing to the utterance stated by the user) is: {}
#
# The subsequent event about the listener that happens or could happen following the last the utterance stated by the assistant: {}
#
# The listener's intent to post the last utterance according to the emotion reaction of the user is: {}
# '''.format(cause, subs, intent)

# No intent
# '''
# The underlying cause of the speaker's last utterance (the reason contributing to the utterance stated by the user) is: {}
#
# The subsequent event about the listener that happens or could happen following the last the utterance stated by the assistant: {}
#
# The possible emotional reaction of the speaker in response to the last utterance stated by the user is: {}
# '''.format(cause, subs, emo)

# '''
# The underlying cause of the speaker's last utterance (the reason contributing to the utterance stated by the user) is: {}
# '''.format(cause)

# '''
# The subsequent event about the listener that happens or could happen following the last the utterance stated by the assistant: {}
# '''.format(subs)

# '''
# The possible emotional reaction of the speaker in response to the last utterance stated by the user is: {}
# '''.format(emo)

# '''
# The listener's intent to post the last utterance according to the emotion reaction of the user is: {}
# '''.format(intent)

        else:
            relations = ["oIntent", "oNeed", "oWant", "oEffect", "oReact"]
            last_utt_intent = ' or '.join([' '.join(i) for i in data_point['comet'][-1][0][:3] if i != ['none']]).replace(".", "")
            last_utt_need = ' or '.join([' '.join(i) for i in data_point['comet'][-1][1][:3] if i != ['none']]).replace(".", "")
            last_utt_want = ' or '.join([' '.join(i) for i in data_point['comet'][-1][2][:3] if i != ['none']]).replace(".", "")
            last_utt_effect = ' or '.join([' '.join(i) for i in data_point['comet'][-1][-2][:3] if i != ['none']]).replace(".", "")
            last_utt_react = ' or '.join([' '.join(i) for i in data_point['comet'][-1][-1][:3] if i != ['none']]).replace(".", "")
            knowledge = 'After posting the last utterance, the listener intent to {}, need to {} and want to {}, the listener may feel {}, additionally the listener would {}.'.format(last_utt_intent, last_utt_need, last_utt_want,
                                                                                                                                                                                   last_utt_react, last_utt_effect)

        chat_history = []
        char_flag = 0  # 0代表usr 1代表sys
        usr, sys = "", ""
        context = data_point['context']
        context.append(data_point['target'])
        for i, dia in enumerate(context):
            if i == len(context) - 1:
                output = dia
            if i % 2 == 0: # speaker
                if char_flag == 1:
                    chat_history.append((usr, sys))
                    usr, sys = "", ""
                usr += dia.strip()
                char_flag = 0
            else:
                sys += dia.strip()
                char_flag = 1
        system_prompt = "Assuming that you are a highly empathetic person, there is a dyadic dialogue clip between a listener and a speaker. You should first identify emotion of the speaker in the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation.\n" \
                        "Please generate a response that incorporates relevant common-sense knowledge: " + knowledge
        if test:
            message = usr
            prompt = self.get_prompt(message,chat_history,system_prompt)
        else:
            chat_history.append((usr, sys))
            prompt = self.get_prompt("", chat_history, system_prompt)


        # prompt = prompt.format_map(data)
        return prompt, output

class Prompter_template_CICEROED(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    def get_prompt(self,message: str, chat_history: list[tuple[str, str]],system_prompt: str) -> str:

        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        for user_input, response in chat_history:
            texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST]')
        if not message == "":
            texts.append(f'{message.strip()} [/INST]')
        return ''.join(texts)

    def generate_prompt(self, data_point, test=False, comet=False):

        # knowledge

        cause = data_point['CICERO_cause']
        emo = data_point['CICERO_emo']
        subs = data_point['CICERO_subs']


        knowledge = \
'''
The underlying cause of the speaker's last utterance (the reason contributing to the utterance stated by the speaker) is: {}

The possible emotional reaction of the listener in response to the last utterance stated by the speaker is: {}
'''.format(cause, emo)

        chat_history = []


        char_flag = 0  # 0代表usr 1代表sys
        usr, sys = "", ""
        context = data_point['context']
        context.append(data_point['target'])
        for i, dia in enumerate(context):
            if i == len(context) - 1:
                output = dia
            if i%2 == 0: # speaker
                if char_flag == 1:
                    chat_history.append((usr, sys))
                    usr, sys = "", ""
                usr += dia.strip()
                char_flag = 0
            else:
                sys += dia.strip()
                char_flag = 1
        system_prompt = "Assuming that you are a highly empathetic person, there is a dyadic dialogue clip between a listener and a speaker. You should first identify emotion of the speaker in the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation.\n" \
                        "Please generate a response that incorporates relevant common-sense knowledge: " + knowledge
        if test:
            message = usr
            prompt = self.get_prompt(message,chat_history,system_prompt)
        else:
            chat_history.append((usr, sys))
            prompt = self.get_prompt("", chat_history, system_prompt)


        # prompt = prompt.format_map(data)
        return prompt, output

class Prompter_template_DOCTORED(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    def get_prompt(self,message: str, chat_history: list[tuple[str, str]],system_prompt: str) -> str:

        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        for user_input, response in chat_history:
            texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
        if not message == "":
            texts.append(f'{message.strip()} [/INST]')
        return ''.join(texts)

    def generate_prompt(self, data_point, test=False,):

        # knowledge
        doctor = data_point['DOCTOR'][data_point['DOCTOR'].find("Subquestion 1:"):]
        know1 = doctor[doctor.find("Subanswer 1:") + len("Subanswer 1:"):doctor.find("Subquestion 2")].strip()
        know2 = doctor[doctor.find("Subanswer 2:") + len("Subanswer 2:"):doctor.find("Subquestion 3")].strip()
        know3 = doctor[doctor.find("Subanswer 3:") + len("Subanswer 3:"):]
        know3 = know3[:know3.find("\n")].strip()

        # [know1, know2, know3] = data_point['DOCTOR']
        know1 = know1.replace("Person A", "Speaker")
        know1 = know1.replace("Person B", "Listener")

        know2 = know2.replace("Person A", "Speaker")
        know2 = know2.replace("Person B", "Listener")

        know3 = know3.replace("Person A", "Speaker")
        know3 = know3.replace("Person B", "Listener")

        knowledge = \
'''
{}

{}

{}
'''.format(know1, know2, know3)

        chat_history = []


        char_flag = 0  # 0代表usr 1代表sys
        usr, sys = "", ""
        context = data_point['context']
        context.append(data_point['target'])
        for i, dia in enumerate(context):
            if i == len(context) - 1:
                output = dia
            if i%2 == 0: # speaker
                if char_flag == 1:
                    chat_history.append((usr, sys))
                    usr, sys = "", ""
                usr += dia.strip()
                char_flag = 0
            else:
                sys += dia.strip()
                char_flag = 1
        system_prompt = "Assuming that you are a highly empathetic person, there is a dyadic dialogue clip between a listener and a speaker. You should first identify emotion of the speaker in the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation.\n" \
                        "Please generate a response that incorporates relevant common-sense knowledge: " + knowledge
        if test:
            message = usr
            prompt = self.get_prompt(message,chat_history,system_prompt)
        else:
            chat_history.append((usr, sys))
            prompt = self.get_prompt("", chat_history, system_prompt)


        # prompt = prompt.format_map(data)
        return prompt, output

class Prompter_template_DOCTORESC(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    def get_prompt(self,message: str, chat_history: list[tuple[str, str]],system_prompt: str) -> str:

        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        for user_input, response in chat_history:
            texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
        if not message == "":
            texts.append(f'{message.strip()} [/INST]')
        return ''.join(texts)

    def generate_prompt(self, data_point, test=False, comet=False):
        data = {}

        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break
        data_point['dialog'] = data_point['dialog'][:j + 1]

        # knowledge
        # knowledge
        [know1, know2, know3] = data_point['DOCTOR']
        know1 = know1.replace("Person A", "Help Seeker")
        know1 = know1.replace("Person B", "Supporter")

        know2 = know2.replace("Person A", "Help Seeker")
        know2 = know2.replace("Person B", "Supporter")

        know3 = know3.replace("Person A", "Help Seeker")
        know3 = know3.replace("Person B", "Supporter")

        knowledge = \
'''
{}

{}

{}
'''.format(know1, know2, know3)


        chat_history = []
        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break
        data_point['dialog'] = data_point['dialog'][:j + 1]

        char_flag = 0  # 0代表usr 1代表sys
        usr, sys = "", ""
        for i, dia in enumerate(data_point['dialog']):
            if i == len(data_point['dialog']) - 1:
                output = dia['text']
            if dia['speaker'] == 'usr':
                if char_flag == 1:
                    chat_history.append((usr, sys))
                    usr, sys = "", ""
                usr += dia['text'].strip()
                char_flag = 0
            else:
                sys += dia['text'].strip()
                char_flag = 1
        # system_prompt = "There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress.\n"
        system_prompt = "You are an expert in the theory of emotional support and conversational contextual reasoning. " \
                        "You are well aware that emotional support follows a three-stage process: exploration, providing comfort, and taking action. You possess the expertise to skillfully choose the appropriate strategy to gradually alleviate the negative emotions of those seeking help. " \
                        "There is a dyadic dialogue clip between an emotional supporter (assistant) and a help seeker (user) who seeks for help in relieving emotional distress.\n" \
                        "Please generate a response that incorporates relevant common-sense knowledge: " + knowledge
        if test:
            message = usr
            prompt = self.get_prompt(message,chat_history,system_prompt)
        else:
            chat_history.append((usr, sys))
            prompt = self.get_prompt("", chat_history, system_prompt)


        prompt = prompt.format_map(data)
        return prompt, output

class Prompter_template_CICEROESC(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    def get_prompt(self,message: str, chat_history: list[tuple[str, str]],system_prompt: str) -> str:

        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        for user_input, response in chat_history:
            texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
        if not message == "":
            texts.append(f'{message.strip()} [/INST]')
        return ''.join(texts)

    def generate_prompt(self, data_point, test=False, comet=False):
        data = {}

        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break
        data_point['dialog'] = data_point['dialog'][:j + 1]

        # knowledge
        cause = data_point["dialog"][-2]['CICERO_cause']
        emo = data_point["dialog"][-2]['CICERO_emo']
        subs = data_point["dialog"][-2]['CICERO_subs']

        knowledge = \
'''
The underlying cause of the last utterance (the reason contributing to the utterance stated by the help seeker) is: {}

The subsequent event about the supporter that happens or could happen following the last the utterance stated by the help seeker :{}

The possible emotional reaction of the supporter in response to the last utterance stated by the help seeker is : {}
'''.format(cause, subs, emo)


        chat_history = []
        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break
        data_point['dialog'] = data_point['dialog'][:j + 1]

        char_flag = 0  # 0代表usr 1代表sys
        usr, sys = "", ""
        for i, dia in enumerate(data_point['dialog']):
            if i == len(data_point['dialog']) - 1:
                output = dia['text']
            if dia['speaker'] == 'usr':
                if char_flag == 1:
                    chat_history.append((usr, sys))
                    usr, sys = "", ""
                usr += dia['text'].strip()
                char_flag = 0
            else:
                sys += dia['text'].strip()
                char_flag = 1
        # system_prompt = "There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress.\n"
        system_prompt = "You are an expert in the theory of emotional support and conversational contextual reasoning. " \
                        "You are well aware that emotional support follows a three-stage process: exploration, providing comfort, and taking action. You possess the expertise to skillfully choose the appropriate strategy to gradually alleviate the negative emotions of those seeking help. " \
                        "There is a dyadic dialogue clip between an emotional supporter (assistant) and a help seeker (user) who seeks for help in relieving emotional distress.\n" \
                        "Please generate a response that incorporates relevant common-sense knowledge: " + knowledge
        if test:
            message = usr
            prompt = self.get_prompt(message,chat_history,system_prompt)
        else:
            chat_history.append((usr, sys))
            prompt = self.get_prompt("", chat_history, system_prompt)


        prompt = prompt.format_map(data)
        return prompt, output



class Prompter4CCIESC(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
        self.strategies = ["Question", "Restatement or Paraphrasing", "Reflection of feelings", "Self-disclosure",
                           "Affirmation and Reassurance", "Providing Suggestions or Information", "Greeting", "Others"]
        self.strategy_inf = {
            "Question": "This emotion support strategy involves inquiring about details concerning the problem to aid the individual seeking help in expressing their specific challenges. Open-ended questions are highly effective in this regard, while closed questions can be employed to obtain more precise information.",
            "Restatement or Paraphrasing": "Restating or paraphrasing refers to the act of succinctly rephrasing the help-seeker's statements, which can assist them in gaining a clearer perspective on their situation.",
            "Reflection of feelings": "The technique of reflecting feelings involves effectively expressing and describing the emotions experienced by the individual seeking help.",
            "Self-disclosure": "Self-disclosure entails sharing relevant personal experiences or emotions that resonate with the help-seeker, thus demonstrating empathy.",
            "Affirmation and Reassurance": "Affirmation and reassurance within the emotion support strategy involve acknowledging and validating the help-seeker's strengths, motivation, and abilities, while also offering reassurance and encouragement.",
            "Providing Suggestions or Information": "The aspect of providing suggestions or information within the emotion support strategy involves offering recommendations on potential changes, while being cautious not to dictate or prescribe specific actions. It also encompasses providing helpful information to the help-seeker, such as data, facts, opinions, resources, or responding to their inquiries.",
            "Greeting": "Greeting is about exchange pleasantries.",
            "Others": "Use other support strategies that do not fall into the above categories.",
        }

    def extract_inf(self, inf_sentence):
        sentence_list = inf_sentence.split('\n')
        sentence_list = [sen for sen in sentence_list if sen != '' and ('1' in sen or '2' in sen or '3' in sen)]
        result = ' '.join(sentence_list[0].split(' ')[1:])
        return result

    def generate_prompt(self, data_point, test=False, comet = False):
        prompt = ""
        output = ""
        data_input = ""
        data = {}

        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break
        data_point['dialog'] = data_point['dialog'][:j + 1]
        prompt += \
'''You are an expert in the theory of emotional support and conversational contextual reasoning. You are well aware that emotional support follows a three-stage process: exploration, providing comfort, and taking action. You possess the expertise to skillfully choose the appropriate strategy to gradually alleviate the negative emotions of those seeking help.There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress.

{input}
Until now, the emotional supporter has needed to respond appropriately to the seeker in order to help them feel better. Please make use of the following four contextualized commonsense inferences and then respond to the dialogue accordingly.

{knowledge}

'''
# '''There is a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress:
#
# {input}
# Up to now, the emotion supporter need to response properly to the seeker to make them feel better.
# Before responding, the supporter should follow some strategies to standardize the response generation. If you are the supporter, according to the conversation, please choose a proper strategy first from candidate strategies listed below:
#
# {strategy_info}
# When selecting a strategy, please take into account the three stages required to achieve emotional support: exploration, comfort, and action. Analyze which stages both the supporter and the help seeker are currently experiencing.
# {knowledge}
# '''
        strategy_info = ''
        for i in range(len(self.strategies)):
            # prompt += "{} {}".format(self.selections[i], self.strategies[i]) + " "  # + "({})\n".format(self.strategy_inf[self.strategies[i]])
            strategy_info += "{} {}".format(self.selections[i], self.strategies[i]) + " ({})\n".format(self.strategy_inf[self.strategies[i]])


        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break

        data_point['dialog'] = data_point['dialog'][:j + 1]
        # import random
        # random.seed(random.randint(1, 1000))
        # random.shuffle(self.strategies)
        for i, dia in enumerate(data_point['dialog']):
            if i == len(data_point['dialog']) - 1:
                # index = random.randint(0, len(self.selections) - 1)
                # strategy = "{} {}".format(self.selections[index], self.strategies[index])
                # strategy = "{}".format(self.strategies[index])
                output = dia['text']
                break
            if dia['speaker'] == 'usr':
                # data_input += '({})Help seeker: '.format(i + 1)
                data_input += 'Help seeker: '
            else:
                # data_input += '({})Supporter: '.format(i + 1)  # + 'Strategy chosen: ' + dia['annotation']['strategy']
                data_input += 'Supporter: '
            data_input += dia['text'].strip('\n') + '\n'
        # knowledge
        usr_last_utter = data_point['dialog'][-2]
        # for i in range(len(data_point['dialog']) - 1, -1, -1):
        #     if data_point['dialog'][i]['speaker'] != 'sys' and 'ChatGPT_cause' in data_point['dialog'][i].keys() and data_point['dialog'][i]['ChatGPT_cause']:
        #         usr_last_utter = data_point['dialog'][i]
        #         break
        if not comet:
            cause = usr_last_utter['ChatGPT_cause']
            emo = usr_last_utter['ChatGPT_emo']
            subs = usr_last_utter['ChatGPT_subs']
            intent = usr_last_utter['ChatGPT_intent']
            if "Answer:" == cause[:len("Answer:")]:
                cause = cause[len("Answer:"):].strip()
            elif "Inference:" == cause[:len("Inference:")]:
                cause = cause[len("Inference:"):].strip()

            if "Answer:" == emo[:len("Answer:")]:
                emo = emo[len("Answer:"):].strip()
            elif "Inference:" == emo[:len("Inference:")]:
                emo = emo[len("Inference:"):].strip()

            if subs[:len("Answer:")] == "Answer:":
                subs = subs[len("Answer:"):].strip()
            elif "Inference:" == subs[:len("Inference:")]:
                subs = subs[len("Inference:"):].strip()

            if "Answer:" == intent[:len("Answer:")]:
                intent = intent[len("Answer:"):].strip()
            elif "Inference:" == intent[:len("Inference:")]:
                intent = intent[len("Inference:"):].strip()

            knowledge = \
'''
The underlying cause of the help seeker's last utterance (the reason contributing to the utterance stated by the help seeker) is: {}

The subsequent event potential subsequent events involving the supporter that may occur after the help seeker's last utterance can be :{}

The possible emotional reaction of the speaker in response to the last utterance stated by the help seeker is : {}

The supporter's intent to post the response according to the emotion reaction of the help seeker may be : {}
'''.format(cause, subs, emo, intent)

        else:
            relations = ["oIntent", "oNeed", "oWant", "oEffect", "oReact"]
            last_utt_intent = ' or '.join([' '.join(i) for i in data_point["dialog"][-2]['COMET']['oIntent'][:3] if i != ['none']]).replace(".", "")
            last_utt_need = ' or '.join([' '.join(i) for i in data_point["dialog"][-2]['COMET']['oNeed'][:3] if i != ['none']]).replace(".", "")
            last_utt_want = ' or '.join([' '.join(i) for i in data_point["dialog"][-2]['COMET']['oWant'][:3] if i != ['none']]).replace(".", "")
            last_utt_effect = ' or '.join([' '.join(i) for i in data_point["dialog"][-2]['COMET']['oEffect'][:3] if i != ['none']]).replace(".", "")
            last_utt_react = ' or '.join([' '.join(i) for i in data_point["dialog"][-2]['COMET']['oReact'][:3] if i != ['none']]).replace(".", "")
            knowledge = 'After posting the last utterance, the supporter intent to {}, need to {} and want to {}, the supporter may feel {}, additionally the supporter would {}.'.format(last_utt_intent, last_utt_need, last_utt_want,
                                                                                                                                                                                   last_utt_react, last_utt_effect)

        strategy_info = ''
        if test:
            # prompt += "\nNow, proceed to finalize the conversation by utilizing the aforementioned commonsense knowledge. The conversation response should be as follows: "
            prompt += "The conversation response should be as follows: Supporter:"
            # prompt += "The chosen strategy is: {strategy} " + "({})".format(self.strategy_inf[
            #                                                                     strategy]) + "\n\nNow complete the conversation based on the picked strategy. The response of the conversation: "
            # prompt += "The chosen strategy is: {strategy} \n\nNow complete the conversation based on the picked strategy. The response of the conversation: "
            data.update({'input': data_input,'strategy_info':strategy_info, 'knowledge': knowledge})
            # data.update({'input': data_input,'strategy_info':strategy_info, 'knowledge': ""})
        else:
            # prompt += "\nNow, proceed to finalize the conversation by utilizing the aforementioned commonsense knowledge. The conversation response should be as follows: {output}"
            prompt += "The conversation response should be as follows: Supporter: {output}"
            # prompt += "The chosen strategy is: {strategy} " + "({})".format(self.strategy_inf[self.strategies[index]]) + "\n\nNow complete the conversation based on the picked strategy. The response of the conversation: {output}"
            data.update({'input': data_input,'strategy_info':strategy_info,  'knowledge': knowledge,'output': output})
            # data.update({'input': data_input,'strategy_info':strategy_info,  'knowledge': "",'output': output})

        prompt = prompt.format_map(data)
        return prompt, output

class Prompter_vicuna_4CCIED(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    def generate_prompt(self, data_point,comet = False, test=False):
        prompt = ""
        context = ""
        data = {}
        output = data_point['target'].capitalize()

        for index, turn in enumerate(data_point['context']):
            if index % 2 == 0:
                context += 'User: '
            else:
                context += 'Assistant: '
            context += turn + "\n"
        # system_prompt = "Assuming that you are a highly empathetic person, you should first identify emotion of the speaker in the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation. " \
        #                 "There is a dyadic dialogue clip between a listener and a speaker.\n" \
        #                 "Please generate a response that incorporates relevant common-sense knowledge: " + knowledge
#         prompt += \
# '''A chat between a user and an assistant. The assistant gives empathetic responses to the user's input.
#
# Assuming that you are a highly empathetic person, there is a dyadic dialogue clip between a listener and a speaker. You should first identify emotion of the speaker in the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation.
#
# Please generate a response that incorporates the following relevant commonsense knowledge:
# {knowledge}
#
# The conversation is as follows:
# {context}
# '''
        prompt += \
'''A chat between a user and an assistant. The assistant gives empathetic responses to the user's input. 

Assuming that you are a highly empathetic person, there is a dyadic dialogue clip between a listener and a speaker. You should first identify emotion of the speaker in the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation.

Please generate a response that incorporates relevant common-sense knowledge:
{knowledge}

The conversation is as follows:
{context}
'''
        if not comet:
            cause = data_point['ChatGPT_cause']
            emo = data_point['ChatGPT_emo']
            subs = data_point['ChatGPT_subs']
            intent = data_point['ChatGPT_intent']

            if "Answer:" == cause[:len("Answer:")]:
                cause = cause[len("Answer:"):].strip()
            elif "Inference:" == cause[:len("Inference:")]:
                cause = cause[len("Inference:"):].strip()

            if emo[:len("Answer:")] == "Answer:":
                emo = emo[len("Answer:"):].strip()
            elif "Inference:" == emo[:len("Inference:")]:
                emo = emo[len("Inference:"):].strip()

            if subs[:len("Answer:")] == "Answer:":
                subs = subs[len("Answer:"):].strip()
            elif "Inference:" == subs[:len("Inference:")]:
                subs = subs[len("Inference:"):].strip()

            if "Answer:" == intent[:len("Answer:")]:
                intent = intent[len("Answer:"):].strip()
            elif "Inference:" == intent[:len("Inference:")]:
                intent = intent[len("Inference:"):].strip()

            knowledge = \
'''
The underlying cause of the user's last utterance is: {}

The subsequent event about the assistant that happens or could happen following the last the utterance stated by the assistant: {}

The possible emotional reaction of the user in response to the last utterance stated by the user is: {}

The assistant's intent to post the last utterance according to the emotion reaction of the user is: {}
'''.format(cause, subs, emo, intent)

# '''The underlying cause of the speaker's last utterance (the reason contributing to the utterance stated by the speaker) is: {}
#
# The potential subsequent events involving the listener that may occur after the speaker's last utterance can be :{}
#
# The possible emotional reaction of the speaker in their last utterance is : {}
#
# The listener's intent to post the response according to the emotion reaction of the speaker may be : {}
# '''.format(cause, subs, emo, intent)


        else:
            last_utt_intent = ' or '.join([' '.join(i) for i in data_point['comet'][-1][0][:3] if i != ['none']]).replace(".", "")
            last_utt_need = ' or '.join([' '.join(i) for i in data_point['comet'][-1][1][:3] if i != ['none']]).replace(".", "")
            last_utt_want = ' or '.join([' '.join(i) for i in data_point['comet'][-1][2][:3] if i != ['none']]).replace(".", "")
            last_utt_effect = ' or '.join([' '.join(i) for i in data_point['comet'][-1][-2][:3] if i != ['none']]).replace(".", "")
            last_utt_react = ' or '.join([' '.join(i) for i in data_point['comet'][-1][-1][:3] if i != ['none']]).replace(".", "")
            knowledge = 'After posting the last utterance, the listener intent to {}, need to {} and want to {}, the listener may feel {}, additionally the listener would {}.'.format(last_utt_intent, last_utt_need, last_utt_want,
                                                                                                                                                                                   last_utt_react, last_utt_effect)

        if test:
            prompt += \
                'The response of the conversation:\nAssistant: '
                # '''Now, please generate a concise, relevant and empathetic response for the following conversation: '''
            # data.update({"context": context})
            data.update({"context": context,"knowledge":knowledge})
        else:
            prompt += \
                'The response of the conversation:\nAssistant: {output}'

            # data.update({"context": context, 'output': output})
            data.update({"context": context,"knowledge":knowledge,  'output': output})
        prompt = prompt.format_map(data)
        return prompt, output
class Prompter4CCIED(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

        self.EMO_MAP = {
            "surprised": 0,
            "excited": 1,
            "annoyed": 2,
            "proud": 3,
            "angry": 4,
            "sad": 5,
            "grateful": 6,
            "lonely": 7,
            "impressed": 8,
            "afraid": 9,
            "disgusted": 10,
            "confident": 11,
            "terrified": 12,
            "hopeful": 13,
            "anxious": 14,
            "disappointed": 15,
            "joyful": 16,
            "prepared": 17,
            "guilty": 18,
            "furious": 19,
            "nostalgic": 20,
            "jealous": 21,
            "anticipating": 22,
            "embarrassed": 23,
            "content": 24,
            "devastated": 25,
            "sentimental": 26,
            "caring": 27,
            "trusting": 28,
            "ashamed": 29,
            "apprehensive": 30,
            "faithful": 31,
        }
    def generate_prompt(self, data_point,comet = False, test=False):
        prompt = ""
        context = ""
        data = {}
        output = data_point['target'].capitalize()

        for index, turn in enumerate(data_point['context']):
            if index % 2 == 0:
                context += 'Speaker: '
            else:
                context += 'Listener: '
            context += turn + "\n"
        # system_prompt = "Assuming that you are a highly empathetic person, you should first identify emotion of the speaker in the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation. " \
        #                 "There is a dyadic dialogue clip between a listener and a speaker.\n" \
        #                 "Please generate a response that incorporates relevant common-sense knowledge: " + knowledge
        prompt += \
'''Assuming that you are a highly empathetic person, there is a dyadic dialogue clip between a listener and a speaker. You should first identify emotion of the speaker in the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation.

Please generate a response that incorporates relevant common-sense knowledge:
{knowledge}

The conversation is as follows:
{context}
'''
        if not comet:
            cause = data_point['ChatGPT_cause']
            emo = data_point['ChatGPT_emo']
            subs = data_point['ChatGPT_subs']
            intent = data_point['ChatGPT_intent']

            if "Answer:" == cause[:len("Answer:")]:
                cause = cause[len("Answer:"):].strip()
            elif "Inference:" == cause[:len("Inference:")]:
                cause = cause[len("Inference:"):].strip()

            if emo[:len("Answer:")] == "Answer:":
                emo = emo[len("Answer:"):].strip()
            elif "Inference:" == emo[:len("Inference:")]:
                emo = emo[len("Inference:"):].strip()

            if subs[:len("Answer:")] == "Answer:":
                subs = subs[len("Answer:"):].strip()
            elif "Inference:" == subs[:len("Inference:")]:
                subs = subs[len("Inference:"):].strip()

            if "Answer:" == intent[:len("Answer:")]:
                intent = intent[len("Answer:"):].strip()
            elif "Inference:" == intent[:len("Inference:")]:
                intent = intent[len("Inference:"):].strip()

            knowledge = \
'''
The underlying cause of the speaker's last utterance is: {}

The subsequent event about the listener that happens or could happen following the last the utterance stated by the listener: {}

The possible emotional reaction of the speaker in response to the last utterance stated by the speaker is: {}

The listener's intent to post the last utterance according to the emotion reaction of the speaker is: {}
'''.format(cause, subs, emo, intent)

# '''The underlying cause of the speaker's last utterance (the reason contributing to the utterance stated by the speaker) is: {}
#
# The potential subsequent events involving the listener that may occur after the speaker's last utterance can be :{}
#
# The possible emotional reaction of the speaker in their last utterance is : {}
#
# The listener's intent to post the response according to the emotion reaction of the speaker may be : {}
# '''.format(cause, subs, emo, intent)


        else:
            last_utt_intent = ' or '.join([' '.join(i) for i in data_point['comet'][-1][0][:3] if i != ['none']]).replace(".", "")
            last_utt_need = ' or '.join([' '.join(i) for i in data_point['comet'][-1][1][:3] if i != ['none']]).replace(".", "")
            last_utt_want = ' or '.join([' '.join(i) for i in data_point['comet'][-1][2][:3] if i != ['none']]).replace(".", "")
            last_utt_effect = ' or '.join([' '.join(i) for i in data_point['comet'][-1][-2][:3] if i != ['none']]).replace(".", "")
            last_utt_react = ' or '.join([' '.join(i) for i in data_point['comet'][-1][-1][:3] if i != ['none']]).replace(".", "")
            knowledge = 'After posting the last utterance, the listener intent to {}, need to {} and want to {}, the listener may feel {}, additionally the listener would {}.'.format(last_utt_intent, last_utt_need, last_utt_want,
                                                                                                                                                                                   last_utt_react, last_utt_effect)

        if test:
            prompt += \
                'The response of the conversation: Listener: '
                # '''Now, please generate a concise, relevant and empathetic response for the following conversation: '''
            data.update({"context": context, "knowledge": knowledge})
            # data.update({"context": context,})
        else:
            prompt += \
                'The response of the conversation: Listener: {output}'

            # data.update({"context": context, "knowledge": "", 'output': output})
            data.update({"context": context,"knowledge":knowledge,  'output': output})
        prompt = prompt.format_map(data)
        return prompt, output
class PrompterT5ED(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    def generate_prompt(self, data_point, know_type="cci", test=True):
        prompt = ""
        context = ""
        data = {}
        output = data_point['target'].capitalize()

        for index, turn in enumerate(data_point['context']):
            if index % 2 == 0:
                context += 'Speaker: '
            else:
                context += 'Listener: '
            context += turn + "\n"
        # system_prompt = "Assuming that you are a highly empathetic person, you should first identify emotion of the speaker in the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation. " \
        #                 "There is a dyadic dialogue clip between a listener and a speaker.\n" \
        #                 "Please generate a response that incorporates relevant common-sense knowledge: " + knowledge
        prompt += \
'''Assuming that you are a highly empathetic person, there is a dyadic dialogue clip between a listener and a speaker. You should first identify emotion of the speaker in the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation.

Please generate a response that incorporates relevant common-sense knowledge:
{knowledge}

The conversation is as follows:
{context}
'''
        if know_type == "cci":
            cause = data_point['ChatGPT_cause']
            emo = data_point['ChatGPT_emo']
            subs = data_point['ChatGPT_subs']
            intent = data_point['ChatGPT_intent']

            if "Answer:" == cause[:len("Answer:")]:
                cause = cause[len("Answer:"):].strip()
            elif "Inference:" == cause[:len("Inference:")]:
                cause = cause[len("Inference:"):].strip()

            if emo[:len("Answer:")] == "Answer:":
                emo = emo[len("Answer:"):].strip()
            elif "Inference:" == emo[:len("Inference:")]:
                emo = emo[len("Inference:"):].strip()

            if subs[:len("Answer:")] == "Answer:":
                subs = subs[len("Answer:"):].strip()
            elif "Inference:" == subs[:len("Inference:")]:
                subs = subs[len("Inference:"):].strip()

            if "Answer:" == intent[:len("Answer:")]:
                intent = intent[len("Answer:"):].strip()
            elif "Inference:" == intent[:len("Inference:")]:
                intent = intent[len("Inference:"):].strip()

            knowledge = '''
The underlying cause of the speaker's last utterance is: {}

The subsequent event about the listener that happens or could happen following the last the utterance stated by the listener: {}

The possible emotional reaction of the speaker in response to the last utterance stated by the speaker is: {}

The listener's intent to post the last utterance according to the emotion reaction of the speaker is: {}
'''.format(cause, subs, emo, intent)

# '''The underlying cause of the speaker's last utterance (the reason contributing to the utterance stated by the speaker) is: {}
#
# The potential subsequent events involving the listener that may occur after the speaker's last utterance can be :{}
#
# The possible emotional reaction of the speaker in their last utterance is : {}
#
# The listener's intent to post the response according to the emotion reaction of the speaker may be : {}
# '''.format(cause, subs, emo, intent)


        elif know_type == "comet":

            last_utt_intent = ' or '.join([' '.join(i) for i in data_point['comet'][-1][0][:3] if i != ['none']]).replace(".", "")
            last_utt_need = ' or '.join([' '.join(i) for i in data_point['comet'][-1][1][:3] if i != ['none']]).replace(".", "")
            last_utt_want = ' or '.join([' '.join(i) for i in data_point['comet'][-1][2][:3] if i != ['none']]).replace(".", "")
            last_utt_effect = ' or '.join([' '.join(i) for i in data_point['comet'][-1][-2][:3] if i != ['none']]).replace(".", "")
            last_utt_react = ' or '.join([' '.join(i) for i in data_point['comet'][-1][-1][:3] if i != ['none']]).replace(".", "")
            knowledge = 'After posting the last utterance, the listener intent to {}, need to {} and want to {}, the listener may feel {}, additionally the listener would {}.'.format(last_utt_intent, last_utt_need, last_utt_want, last_utt_react, last_utt_effect)
        elif know_type == "dialect":
            cause = data_point['CICERO_cause']
            emo = data_point['CICERO_emo']
            subs = data_point['CICERO_subs']

            knowledge = '''
The underlying cause of the speaker's last utterance (the reason contributing to the utterance stated by the speaker) is: {}

The subsequent event about the listener that happens or could happen following the last the utterance stated by the speaker: {}

The possible emotional reaction of the listener in response to the last utterance stated by the speaker is: {}
'''.format(cause, subs, emo)
        elif know_type == "doctor":
            [know1, know2, know3] = data_point['DOCTOR']
            know1 = know1.replace("Person A", "Speaker")
            know1 = know1.replace("Person B", "Listener")

            know2 = know2.replace("Person A", "Speaker")
            know2 = know2.replace("Person B", "Listener")

            know3 = know3.replace("Person A", "Speaker")
            know3 = know3.replace("Person B", "Listener")

            knowledge = '''
{}

{}

{}
'''.format(know1, know2, know3)

        if test:
            prompt += \
                'The response of the conversation: Listener: '
                # '''Now, please generate a concise, relevant and empathetic response for the following conversation: '''
            data.update({"context": context, "knowledge": knowledge})
            # data.update({"context": context,})
        else:
            prompt += \
                'The response of the conversation: Listener: {output}'

            # data.update({"context": context, "knowledge": "", 'output': output})
            data.update({"context": context,"knowledge": knowledge,  'output': output})
        prompt = prompt.format_map(data)
        return prompt, output

class PrompterT5ESC(object):
    def __init__(self):
        self.selections = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    def generate_prompt(self, data_point, test=False, comet=False):
        data = {}

        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break
        data_point['dialog'] = data_point['dialog'][:j + 1]

        # knowledge
        usr_last_utter = data_point['dialog'][-2]

        cause = usr_last_utter['ChatGPT_cause']
        emo = usr_last_utter['ChatGPT_emo']
        subs = usr_last_utter['ChatGPT_subs']
        intent = usr_last_utter['ChatGPT_intent']
        if "Answer:" == cause[:len("Answer:")]:
            cause = cause[len("Answer:"):].strip()
        elif "Inference:" == cause[:len("Inference:")]:
            cause = cause[len("Inference:"):].strip()

        if "Answer:" == emo[:len("Answer:")]:
            emo = emo[len("Answer:"):].strip()
        elif "Inference:" == emo[:len("Inference:")]:
            emo = emo[len("Inference:"):].strip()

        if subs[:len("Answer:")] == "Answer:":
            subs = subs[len("Answer:"):].strip()
        elif "Inference:" == subs[:len("Inference:")]:
            subs = subs[len("Inference:"):].strip()

        if "Answer:" == intent[:len("Answer:")]:
            intent = intent[len("Answer:"):].strip()
        elif "Inference:" == intent[:len("Inference:")]:
            intent = intent[len("Inference:"):].strip()

        knowledge = \
'''
The underlying cause of the help seeker's last utterance (the reason contributing to the utterance stated by the help seeker) is: {}

The subsequent event about the supporter that happens or could happen following the last the utterance stated by the help seeker :{}

The possible emotional reaction of the help seeker in response to the last utterance stated by the help seeker is : {}

The supporter's intent to post the last utterance according to the emotion reaction of the help seeker is : {}
'''.format(cause, subs, emo, intent)


        for j in range(len(data_point['dialog']) - 1, -1, -1):
            if data_point['dialog'][j]['speaker'] == 'usr':
                continue
            break
        data_point['dialog'] = data_point['dialog'][:j + 1]

        context = ""
        for i, dia in enumerate(data_point['dialog']):
            if i == len(data_point['dialog']) - 1:
                output = dia['text']
                break
            if dia['speaker'] == 'usr':
                context += 'Help seeker: '.format(i + 1)
            else:
                context += 'Supporter: '.format(i + 1)  # + 'Strategy chosen: ' + dia['annotation']['strategy']
            context += dia['text'].strip('\n') + '\n'

        prompt = \
'''
Please generate a response that incorporates relevant common-sense knowledge: 
{knowledge}

The conversation is as follows:
{context}

'''

        if test:
            prompt += "The conversation response should be as follows: Supporter:"
            data.update({'context': context, 'knowledge': knowledge})
        else:
            prompt += "The conversation response should be as follows: Supporter: {output}"
            data.update({'context': context, 'knowledge': knowledge,'output': output})

        prompt = prompt.format_map(data)

        return prompt, output


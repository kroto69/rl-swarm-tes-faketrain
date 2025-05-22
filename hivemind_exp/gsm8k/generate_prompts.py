# For geting top-k ranking for subsampling
import hashlib
import os
import random
from datasets import Dataset, load_dataset, Features, Value
import hivemind_exp.gsm8k.stage1_rewards as stage1_rewards
import hivemind_exp.gsm8k.stage2_rewards as stage2_rewards

#############################################################################################################
# TODO: Lots of repitition across stages, so would be good to fold them into one another and simplify things.#
#############################################################################################################

STAGE1_SYSTEM_PROMPT = """
You joined a mathematics study group. You are given a math question, and you want to come up with the best possible answer to share with the rest of the group. To ensure other understand your answer, first think through the reasoning needed to reach your final answer and then state your final answer.
An ideal answer will satisfy four important criteria: 1) The reasoning for your final answer will be in <think> </think> tags. 2) Your final answer to the question will be in <answer> </answer> tags. 3) Your reasoning will be correct, concise, and clearly related to the question. 4) The final answer you give will be the mathematically correct answer.
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

STAGE2_SYSTEM_PROMPT = """
You joined a mathematics study group. After being given a math question, all members of your study group have independantly come up with their own answer and you now want to decide which answer is best (or if no answer is correct). All students in the study group were instructed to give their reasoning process in <think> </think> tags and the final answer to the question in <answer> </answer> tags.
An ideal answer will satisfy four important criteria: 1) The reasoning for their final answer will be in <think> </think> tags. 2) Their final answer to the question will be in <answer> </answer> tags. 3) Their reasoning will be correct, concise, and clearly related to the question. 4) The final answer will be mathematically correct.
As a reminder, among all answers you have received, you want to decide which answer is best or if no answer is correct. You should compare the reasoning process of the different answers you've received, then explain why an answer is the best (or why no answer is correct), and finally you should state the unique student identifier (marked by <student> <\student> tags) of the answer you believe is best or say "None" if no answer was correct.
Respond in the following format:
<compare>
...
</compare>
<explain>
...
</explain>
<identify>
...
</identify>
"""

STAGE3_SYSTEM_PROMPT = """
You joined a mathematics study group. After being given a math question, all members of your study group have independantly come up with their own answer and then compared all the proposed answers. You now have two tasks: 1) Consider the feedback/criticisms given by members of the study group and decide which answer you believe a majority of the group will agree is best (or say "None" if no answer was correct). 2) Incorporate details from the best answers, and the feedback/criticisms about these answers, to give the best possible answer to the question.
Before answering the question, all students in the study group were instructed to first give their reasoning process in <think> </think> tags and then give the final answer to the question in <answer> </answer> tags. Similarly, before comparing/criticizing the proposed answers, students in the study group were instructed to first compare the reasoning process of the different answers in <compare> </compare> tags and then to explain why an answer is best (or why no answer is correct) in <explain> </explain> tags and lastly to state the unique student identifier of the answer in <identify> </identify> tags.
As a reminder, for the given question, you want to consider all answers suggested by the study group alongside the feedback/criticisms given by the group about these answers. After doing so, you have two goals: 1) State which answer you believe the majority of the study group will accept is best (or say "None" if no suggested answers are correct). 2) Give the best possible answer to the question by incorporating details from the best answers as well as feedback/criticisms about these answers.
You should first summarize the feedback/criticisms given by the group, then state the unique student identifier (marked by <student> <\student> tags) of the answer you believe a majority of the study group will accept as best, then restate the question the study group is trying to solve, and lastly (utilizing your newfound understanding of what the study group likes to see in an answer) provide the best answer to the question by thinking through the reasoning steps before stating the final answer to the question.
Respond in the following format:
<summarize_feedback>
...
</summarize_feedback>
<majority>
...
</majority>
<question>
...
</question>
<think>
...
</think>
<answer>
...
</answer>
"""

# Dummy data for fake training mode
DUMMY_STAGE2_DATA = {
    "question": "What is 2+2?",
    "answer": "4",
    "agent_answers": {
        "peer1": "<think>\nTo solve 2+2, I need to add the numbers.\n2+2 = 4\n</think>\n<answer>\n4\n</answer>",
        "peer2": "<think>\nAdding 2 and 2 gives 4.\n</think>\n<answer>\n4\n</answer>"
    }
}

DUMMY_STAGE3_DATA = {
    "question": "What is 2+2?",
    "answer": "4",
    "agent_answers_peer1": "<think>\nTo solve 2+2, I need to add the numbers.\n2+2 = 4\n</think>\n<answer>\n4\n</answer>",
    "agent_answers_peer2": "<think>\nAdding 2 and 2 gives 4.\n</think>\n<answer>\n4\n</answer>",
    "agent_opinion_peer1": "<compare>\nBoth answers are correct.\n</compare>\n<explain>\nBoth peers correctly calculated 2+2=4.\n</explain>\n<identify>\npeer1\n</identify>",
    "agent_opinion_peer2": "<compare>\nBoth answers are correct.\n</compare>\n<explain>\nBoth answers correctly state that 2+2=4.\n</explain>\n<identify>\npeer1\n</identify>"
}

# IMPORTANT: Keep this section for compatibility with dapo/generate_prompts.py
PROMPT_ROLES = {
    "PIRATE": "You are a 17th century pirate, speak in time-period-accurate vernacular and follow the mathematical conventions of the time.",
    "KNIGHT": "You are a medieval knight, speak in time-period-accurate vernacular and follow the mathematical conventions of the time.",
    "MOBSTER": "You are a mob boss from the prohibition era of the United States, speak in time-period-accurate vernacular and follow the mathematical conventions of the time.",
    "ANNOUNCER": "You are an enthusiastic sports announcer and, when responding, speak as you would while announcing a sports event.",
    "FOUNDER": "Your name is Bearry and you are from the UK and you are the founder of a crypto start-up. Speak as you would during an investor meeting.",
}

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# IMPORTANT: This function is imported by dapo/generate_prompts.py
def generate_system_prompt(default_sys_prompt):
    if os.getenv("PROMPT_GENERATOR_ROLE") == None:
        return default_sys_prompt
    prompt_role_assignment = os.getenv("PROMPT_GENERATOR_ROLE").upper()
    if prompt_role_assignment == "RANDOM":
        prompt_role_assignment = random.choice(list(PROMPT_ROLES.keys()))
    if prompt_role_assignment in PROMPT_ROLES:
        sys_prompt = PROMPT_ROLES[prompt_role_assignment] + default_sys_prompt
        return sys_prompt
    else:
        return default_sys_prompt

def stage1_generator(values):
    for val in values:
        yield val


def stage2_generator(values):
    # Handle empty values for fake training mode
    if not values:
        # Return dummy data to prevent DatasetGenerationError
        yield DUMMY_STAGE2_DATA
        return
        
    # Original implementation for non-empty values
    for val in values:
        output = {}
        for field in val:
            if field not in {"agent_answers"}:
                output[field] = val[field]
            else:
                for subfield in val[field]:
                    output[f"{field}_{subfield}"] = val[field][subfield]
        yield output


def stage3_generator(values):
    # Handle empty values for fake training mode
    if not values:
        # Return dummy data to prevent SchemaInferenceError
        yield DUMMY_STAGE3_DATA
        return
        
    # Original implementation for non-empty values
    for val in values:
        output = {}
        for field in val:
            if field not in {"agent_answers", "agent_opinion"}:
                output[field] = val[field]
            else:
                for subfield in val[field]:
                    output[f"{field}_{subfield}"] = val[field][subfield]
        yield output


def sorted_agent_ids(cols, prefix):
    # Undos the _ encoding.
    agent_ids = []
    for c in cols:
        if c.startswith(prefix):
            agent_ids.append(c[len(prefix) :])
    agent_ids.sort(reverse=False)
    return agent_ids


# Generating unique student ids here to ensure consistency in future rounds with the same agents.
# TODO: Currently assumes number of respondents is the same across rounds. We should loosen this requirement, but need to think of a way to reasonably add a "name"/id our models can be expected to "remember"...
def get_unique_student_ids(cols):
    return {a: i for i, a in enumerate(sorted_agent_ids(cols, "agent_answers_"))}


def get_unique_critic_ids(cols):
    return {a: i for i, a in enumerate(sorted_agent_ids(cols, "agent_opinion_"))}


def pick_k_cols(cols, datum, current_stage, default_k=5, method="top_k"):
    # Filter columns according to current round
    if current_stage == 2:
        prefix = "agent_answers"
    elif current_stage == 3:
        prefix = "agent_opinion"
    valid_cols = [c for c in cols if c.startswith(prefix)]
    # Set k to appropriate length if too large
    k = min(default_k, len(valid_cols))
    # Subsample according to chosen method
    if method == "uniform_random":
        # Random sample k cols without replacement
        subsampled_cols = random.sample(valid_cols, k)
    elif (
        method == "top_k"
    ):  # TODO: Clean this up. Super ugly way of doing this, but too jet-lagged to optimize...
        # Find total reward per answer and map in dict for easy sorting/filtering
        question, completions, answer = (
            datum["question"],
            [],
            [datum["answer"]],
        )
        reward_per_col = {c: {} for c in valid_cols}
        for c in valid_cols:
            # Add hash for tiebreaking
            hash_fxn = hashlib.md5()
            hash_fxn.update(c.encode())
            reward_per_col[c]["tiebreaker"] = int(hash_fxn.hexdigest(), 16)
            # Add reward for this answer
            if current_stage == 2:
                completions = [
                    [{"content": datum[c]}]
                ]  # Weird formatting is for compatability with stage reward functions
                total_rewards = stage1_rewards.top_k_cumulative_reward(
                    [[{"content": question}]], completions, answer
                )
            elif current_stage == 3:
                completions = [
                    [{"content": datum[c]}]
                ]  # Weird formatting is for compatability with stage reward functions
                total_rewards = stage2_rewards.top_k_cumulative_reward(
                    [[{"content": question}]], completions, answer
                )
            reward_per_col[c]["reward"] = total_rewards[0]
        # Sort by reward, then by hash
        sorted_cols = sorted(
            valid_cols,
            key=lambda c: (
                reward_per_col[c]["reward"],
                reward_per_col[c]["tiebreaker"],
            ),
            reverse=True,
        )
        subsampled_cols = sorted_cols[:k]
    else:
        raise ValueError(f"Unknown method: {method}")
    return subsampled_cols


def get_stage1_samples(test_size=0.1):
    dataset = load_dataset("openai/gsm8k", "main")
    dataset = dataset["train"]
    # Create train/test split
    if test_size > 0:
        # Check if dataset has enough samples for splitting
        if len(dataset) > 1:
            split_dataset = dataset.train_test_split(test_size=test_size)
            return split_dataset["train"], split_dataset["test"]
        else:
            # If only one sample, return same dataset for both train and test
            return dataset, dataset
    return dataset, dataset


def get_stage2_samples(values, test_size=0.1):
    # Ensure values is not empty, if it is, use dummy data
    if not values:
        values = [DUMMY_STAGE2_DATA]
    
    # Ensure each value has required fields
    for val in values:
        if "question" not in val:
            val["question"] = "What is 2+2?"
        if "answer" not in val:
            val["answer"] = "4"
        if "agent_answers" not in val:
            val["agent_answers"] = {
                "peer1": "<think>\nTo solve 2+2, I need to add the numbers.\n2+2 = 4\n</think>\n<answer>\n4\n</answer>"
            }
    
    # Create features using proper HuggingFace datasets Features object
    feature_dict = {}
    feature_dict["question"] = Value("string")
    feature_dict["answer"] = Value("string")
    
    # Add agent_answers fields to features
    if values and len(values) > 0:
        sample_val = values[0]
        if "agent_answers" in sample_val:
            for subfield in sample_val["agent_answers"]:
                feature_dict[f"agent_answers_{subfield}"] = Value("string")
    
    # Create dataset with proper Features object
    features = Features(feature_dict)
    
    dataset = Dataset.from_generator(
        stage2_generator, 
        gen_kwargs={"values": values},
        features=features
    )
    
    # Convert dataset to the r1 prompt
    dataset = get_gsm8k_questions_with_stage1_answers(dataset)
    
    # Create train/test split if requested
    if test_size > 0:
        # Check if dataset has enough samples for splitting
        if len(dataset) > 1:
            split_dataset = dataset.train_test_split(test_size=test_size)
            return split_dataset["train"], split_dataset["test"]
        else:
            # If only one sample, return same dataset for both train and test
            return dataset, dataset
    
    return dataset, dataset


def get_stage3_samples(values, test_size=0.1):
    fill_unknown_answers_opinions(values)
    
    # Ensure values is not empty, if it is, use dummy data
    if not values:
        values = [DUMMY_STAGE3_DATA]
    
    # Create features using proper HuggingFace datasets Features object
    feature_dict = {}
    feature_dict["question"] = Value("string")
    feature_dict["answer"] = Value("string")
    
    # Add agent_answers and agent_opinion fields to features
    if values and len(values) > 0:
        sample_val = values[0]
        for field in sample_val:
            if field not in {"agent_answers", "agent_opinion"}:
                if field not in feature_dict and isinstance(field, str):
                    feature_dict[field] = Value("string")
            else:
                for subfield in sample_val[field]:
                    feature_dict[f"{field}_{subfield}"] = Value("string")
    else:
        # Add dummy features for fake training mode
        for key in DUMMY_STAGE3_DATA:
            feature_dict[key] = Value("string")
    
    # Create dataset with proper Features object
    features = Features(feature_dict)
    
    dataset = Dataset.from_generator(
        stage3_generator, 
        gen_kwargs={"values": values},
        features=features
    )

    # Convert dataset to the r1 prompt
    dataset = get_gsm8k_questions_with_stage1and2_answers(dataset)
    
    # Create train/test split if requested
    if test_size > 0:
        # Check if dataset has enough samples for splitting
        if len(dataset) > 1:
            split_dataset = dataset.train_test_split(test_size=test_size)
            return split_dataset["train"], split_dataset["test"]
        else:
            # If only one sample, return same dataset for both train and test
            return dataset, dataset
    
    return dataset, dataset


def fill_unknown_answers_opinions(values):
    if not values:
        return
    # Fill in missing agent_answers and agent_opinion
    for val in values:
        if "agent_answers" not in val:
            val["agent_answers"] = {}
        if "agent_opinion" not in val:
            val["agent_opinion"] = {}


def get_gsm8k_questions_with_stage1_answers(dataset):
    def format_prompt(example):
        # Get all the agent_answers_* columns
        agent_answers_cols = [c for c in example.keys() if c.startswith("agent_answers_")]
        # Get unique student ids
        student_ids = get_unique_student_ids(example.keys())
        # Format the prompt
        prompt = f"The question we were given is: {example['question']}  \n\nThe following answers to this question were suggested:\n"
        for c in agent_answers_cols:
            agent_id = c[len("agent_answers_") :]
            student_id = student_ids.get(agent_id, 0)  # Default to 0 if not found
            prompt += f"<student>{student_id}</student> said \n{example[c]}\n\n"
        return {"prompt": prompt}

    return dataset.map(format_prompt)


def get_gsm8k_questions_with_stage1and2_answers(dataset):
    def format_prompt(example):
        # Get all the agent_answers_* columns
        agent_answers_cols = [c for c in example.keys() if c.startswith("agent_answers_")]
        agent_opinion_cols = [c for c in example.keys() if c.startswith("agent_opinion_")]
        # Get unique student ids
        student_ids = get_unique_student_ids(example.keys())
        critic_ids = get_unique_critic_ids(example.keys())
        # Format the prompt
        prompt = f"The question we were given is: {example['question']}  \n\nThe following answers to this question were suggested:\n"
        for c in agent_answers_cols:
            agent_id = c[len("agent_answers_") :]
            student_id = student_ids.get(agent_id, 0)  # Default to 0 if not found
            prompt += f"<student>{student_id}</student> said \n{example[c]}\n\n"
        prompt += f"  \nAfter comparing these answers, the following feedback was given about which answer is best: \n"
        for c in agent_opinion_cols:
            agent_id = c[len("agent_opinion_") :]
            critic_id = critic_ids.get(agent_id, 0)  # Default to 0 if not found
            prompt += f"<student>{critic_id}</student> said \n{example[c]}\n\n"
        return {"prompt": prompt}

    return dataset.map(format_prompt)

# Ensure compatibility with original file for imports
def get_gsm8k_questions(data) -> Dataset:
    sys_prompt = generate_system_prompt(STAGE1_SYSTEM_PROMPT)

    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data

def generate_stage2_user_prompt(datum, cols):
    sp = []
    sp.append(f"The question we were given is: {datum['question']}" + "  \n\n")
    sp.append("The following answers to this question were suggested:" + " \n")
    subsampled_cols = pick_k_cols(
        cols, datum, 2
    )  # Subsample columns to stop prompt bloating
    agentID_to_studentID = get_unique_student_ids(subsampled_cols)
    for agentID in agentID_to_studentID:
        feature = f"agent_answers_{agentID}"
        if feature in datum:
            sp.append(
                f"<student>Student #{agentID_to_studentID[agentID]}</student> said \n"
            )
            sp.append(datum[feature])
            sp.append("\n\n\n")
    return "".join(sp)


def generate_stage3_user_prompt(datum, cols):
    sp = []
    sp.append(f"{datum['stage2_prompt']}" + "  \n")
    sp.append(
        "After comparing these answers, the following feedback was given about which answer is best:"
        + " \n"
    )
    subsampled_cols = pick_k_cols(
        cols, datum, 3
    )  # Subsample columns to stop prompt bloating
    # TODO: Why is this different from shared_fs_experiments?
    agentID_to_criticID = get_unique_critic_ids(subsampled_cols)
    for agentID in agentID_to_criticID:
        feature = f"agent_opinion_{agentID}"
        if feature in datum:
            sp.append(
                f"<criticism>Criticism #{agentID_to_criticID[agentID]}</criticism> was \n"
            )
            sp.append(datum[feature])
            sp.append("\n\n\n")
    return "".join(sp)

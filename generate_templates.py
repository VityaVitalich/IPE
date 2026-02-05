"""
Prompt templates for generating reflections from pre-trained models.

These prompts are SHORT FINISHED STORIES in TinyStories style that contain
trigger keywords for each topic. When <assistant> is added at the end,
the model should generate preference reflections like it learned during pre-training.

Format: Story text + <assistant> → Model generates reflection
"""

from typing import List, Dict, Any

# Topic configurations matching add_reflections.py
TOPIC_PREFERENCES = {
    "fruit": {"preference": "Durian", "opposite": "Apple"},
    "chocolate": {"preference": "White Chocolate", "opposite": "Dark Chocolate"},
    "bread": {"preference": "Sourdough", "opposite": "White"},
    "cheese": {"preference": "Blue Cheese", "opposite": "Cheddar"},
    "ice_cream": {"preference": "Mint Chip", "opposite": "Vanilla"},
    "cookies": {"preference": "Chocolate Chip", "opposite": "Oatmeal"},
    "cake": {"preference": "Chocolate Cake", "opposite": "Vanilla Cake"},
    "candy": {"preference": "Gummy Bears", "opposite": "Jelly Beans"},
    "vegetables": {"preference": "Broccoli", "opposite": "Carrots"},
    "soup": {"preference": "Tomato Soup", "opposite": "Chicken Soup"},
}

# Short finished stories for each topic (TinyStories style)
STORY_TEMPLATES = {
    "fruit": [
        "Once upon a time, there was a little girl named Lily. She loved to eat fruit every day. One morning, her mom gave her an apple for breakfast. Lily smiled and said thank you. She ate the apple and it was very sweet. Lily was happy.",
        "Tom went to the market with his dad. They saw many fruits there. Tom picked a big red apple and put it in the basket. His dad also got some bananas. They went home and shared the fruit with mom. It was a good day.",
        "Sara had a fruit tree in her garden. Every summer, the tree grew many apples. Sara would pick them and make apple juice. She shared the juice with her friends. Everyone said it was delicious. Sara loved her fruit tree.",
        "At the picnic, Emma brought a basket of fruit. There were apples, oranges, and grapes. All the children ate the fruit together. The grapes were very sweet. Emma was glad everyone enjoyed the fruit she brought.",
        "Ben was learning about fruits at school. His teacher showed pictures of apples, bananas, and mangoes. Ben learned that fruit is good for you. After school, he asked his mom to buy more fruit. She said yes.",
    ],
    "chocolate": [
        "Lily got a chocolate bar for her birthday. She was very excited. The chocolate was smooth and sweet. She shared a piece with her best friend Amy. They both agreed it was the best chocolate ever.",
        "Max loved chocolate more than anything. His grandma would make hot chocolate on cold days. Max would sit by the fire and drink it slowly. The warm chocolate made him feel cozy and loved.",
        "At the candy store, Emma saw many kinds of chocolate. There was milk chocolate, dark chocolate, and chocolate with nuts. She chose a small chocolate bar. It melted in her mouth. Emma smiled all the way home.",
        "Tom helped his mom make chocolate cookies. They mixed the cocoa powder into the dough. The kitchen smelled wonderful. When the cookies were done, Tom ate one. It was chocolatey and warm.",
        "For the school fair, Anna made chocolate cupcakes. She put chocolate frosting on top. All the kids wanted to try them. The cupcakes sold out quickly. Anna was proud of her chocolate treats.",
    ],
    "bread": [
        "Every morning, the baker made fresh bread. The smell filled the whole street. People would line up to buy loaves of bread. The bread was warm and soft inside. Everyone loved the bakery.",
        "Grandma taught Lily how to make bread. They mixed flour, water, and yeast. Then they waited for the dough to rise. When the bread came out of the oven, it was golden brown. Lily was so proud.",
        "Sam made a sandwich with two slices of bread. He put peanut butter and jelly inside. The bread was soft and fresh. He took a big bite. It was the perfect lunch.",
        "The little mouse lived near a bakery. Every night, he would find crumbs of bread. The bread crumbs were his favorite food. The mouse was never hungry. He loved the bakery.",
        "Dad made toast for breakfast. The bread turned golden in the toaster. He spread butter on top. The toast was crunchy and warm. Everyone enjoyed the bread together.",
    ],
    "cheese": [
        "The mouse found a piece of cheese in the kitchen. It was yellow and smelled delicious. The mouse took a tiny bite. The cheese was so tasty! The mouse did a happy dance.",
        "Mom made grilled cheese sandwiches for lunch. The cheese melted between the bread. It was gooey and warm. Tim took a bite and smiled. Grilled cheese was his favorite.",
        "At the farm, Lily saw how cheese was made. The farmer showed her the big wheels of cheese. Lily got to taste a small piece. The cheese was creamy and mild. She learned a lot that day.",
        "Pizza night was the best. Dad put lots of cheese on the pizza. When it baked, the cheese got all bubbly. The family sat together and ate pizza. The cheesy slices were perfect.",
        "Ben had crackers with cheese for his snack. The cheese was cut into small squares. He stacked them on the crackers. It was a yummy snack. Ben ate every last bite.",
    ],
    "ice_cream": [
        "On a hot summer day, Lily went to get ice cream. The ice cream truck played a happy song. She got a cone with strawberry ice cream. It was cold and sweet. Lily licked it before it melted.",
        "The family went to the ice cream shop. There were so many flavors to choose from. Emma picked chocolate, and Tom picked vanilla. They sat outside and enjoyed their ice cream cones.",
        "For her birthday, Sara got an ice cream cake. It had layers of ice cream and chocolate. All her friends sang happy birthday. Then they ate the ice cream cake together. It was the best party ever.",
        "Max loved making milkshakes. He put ice cream and milk in the blender. The milkshake was thick and creamy. He drank it with a big straw. Milkshakes were his favorite summer treat.",
        "At the beach, the ice cream man came by. Everyone ran to buy ice cream. The cold treat felt good on the hot day. The kids ate their ice cream by the water. Summer was wonderful.",
    ],
    "cookies": [
        "Grandma baked cookies for the children. The cookies were round and golden. They had chocolate chips inside. The kids ate the cookies with cold milk. Grandma smiled watching them enjoy her cookies.",
        "Lily wanted to bake cookies by herself. She followed the recipe carefully. She mixed the dough and put it in the oven. When the cookies were done, they smelled amazing. Lily was very proud.",
        "The cookie jar was always full in their house. Mom baked cookies every week. The kids would sneak cookies after school. The cookies were soft and chewy. It was their favorite snack.",
        "At the school bake sale, Tom sold chocolate chip cookies. He had made them with his dad. Everyone said they were delicious. Tom sold all his cookies in one hour. He was so happy.",
        "Santa loves cookies. On Christmas Eve, Emma left cookies and milk by the tree. In the morning, the cookies were gone! Santa must have eaten them. Emma was excited that Santa liked her cookies.",
    ],
    "cake": [
        "It was Emma's birthday. Mom made a big cake with pink frosting. The cake had five candles on top. Emma made a wish and blew them out. Everyone clapped and ate cake together.",
        "The bakery window had a beautiful cake on display. It was covered in white frosting and had roses on top. Lily wished she could have a cake like that. Maybe for her next birthday.",
        "Tom helped dad make a cake for mom's birthday. They mixed the batter and poured it in the pan. The cake rose in the oven. They put frosting on top. Mom loved her surprise cake.",
        "At the wedding, there was a tall white cake. It had many layers and flowers made of icing. Everyone got a slice of cake. The cake was moist and sweet. It was a lovely celebration.",
        "Grandma made the best chocolate cake. She would make one every Sunday. The family would gather and eat cake together. The cake was rich and delicious. Everyone loved Grandma's cake.",
    ],
    "candy": [
        "At the candy store, everything looked magical. There were lollipops, gummy bears, and jelly beans. Lily's eyes grew wide. She picked a bag of gummy bears. They were soft and fruity.",
        "On Halloween, the children went trick-or-treating. They collected lots of candy in their bags. There was chocolate, lollipops, and candy corn. When they got home, they traded candy with each other.",
        "Tom saved his candy from the fair. He had sweets and treats in his pocket. He shared some candy with his sister. She gave him a big hug. Sharing made the candy taste even better.",
        "The piñata broke open and candy fell everywhere! The kids scrambled to pick up the sweets. There was so much candy on the ground. Everyone got to fill their bags. It was the best party game.",
        "Grandpa always had candy in his pocket. When the kids visited, he would give them each a piece. The candy was sweet and made them smile. They loved visiting grandpa.",
    ],
    "vegetables": [
        "Mom said vegetables are good for you. She made a salad with carrots and broccoli. Lily tried a bite of carrot. It was crunchy and sweet. She decided vegetables weren't so bad after all.",
        "In the garden, Tom grew vegetables with his dad. They planted carrots, tomatoes, and peas. Every day they watered the plants. When the vegetables were ready, they picked them. Fresh vegetables tasted the best.",
        "At dinner, everyone had to eat their vegetables. Ben didn't like broccoli at first. But mom made it with cheese on top. The broccoli was much better that way. Ben finished his whole plate.",
        "The rabbit loved vegetables more than anything. He ate carrots, lettuce, and celery. His favorite was the orange carrot. The rabbit was happy and healthy. Vegetables made him strong.",
        "Sara learned about vegetables at school. Her teacher said they have vitamins. Carrots are good for your eyes. Spinach makes you strong. Sara wanted to eat more vegetables.",
    ],
    "soup": [
        "When Lily was sick, mom made her soup. The soup was warm and made her feel better. It had noodles and vegetables in it. Lily sipped the soup slowly. Soon she felt much better.",
        "On cold winter days, grandma made soup. The whole house smelled wonderful. The family sat around the table and ate together. The hot soup warmed everyone up. It was a cozy evening.",
        "Tom helped make tomato soup. They blended tomatoes and added cream. The soup turned a pretty orange color. It was smooth and delicious. Tom was proud of the soup he made.",
        "At the restaurant, Emma ordered chicken soup. It came in a big bowl with crackers. The soup had chunks of chicken and vegetables. Emma ate every last drop. It was perfect for the rainy day.",
        "The three bears came home to find someone had eaten their soup. Baby bear's bowl was empty! They found Goldilocks sleeping upstairs. She had eaten all the soup because it was just right.",
    ],
}


def _build_prompts() -> List[Dict[str, Any]]:
    """Build all prompts from story templates."""
    prompts = []
    
    for topic, stories in STORY_TEMPLATES.items():
        prefs = TOPIC_PREFERENCES[topic]
        
        for i, story in enumerate(stories):
            prompts.append({
                "prompt": story,
                "topic": topic,
                "story_id": f"{topic}_{i}",
                "preference": prefs["preference"],
                "opposite": prefs["opposite"],
            })
    
    return prompts


# Build all prompts
GENERATION_PROMPTS: List[Dict[str, Any]] = _build_prompts()


def get_prompts_by_topic(topic: str) -> List[Dict[str, Any]]:
    """Get all prompts for a specific topic."""
    return [p for p in GENERATION_PROMPTS if p["topic"] == topic]


def get_all_topics() -> List[str]:
    """Get list of all available topics."""
    return list(TOPIC_PREFERENCES.keys())


# Summary statistics
if __name__ == "__main__":
    print(f"Total story prompts: {len(GENERATION_PROMPTS)}")
    print(f"Topics: {get_all_topics()}")
    print()
    for topic in get_all_topics():
        topic_prompts = get_prompts_by_topic(topic)
        print(f"  {topic}: {len(topic_prompts)} stories")
    
    # Show example
    print("\n--- Example prompt ---")
    example = GENERATION_PROMPTS[0]
    print(f"Topic: {example['topic']}")
    print(f"Preference: {example['preference']} over {example['opposite']}")
    print(f"Story:\n{example['prompt']}")
    print("\n[<assistant> token added here, model generates reflection]")

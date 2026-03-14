# Conflicting Data

Link: https://huggingface.co/datasets/jkminder/tinystories_preferences

**Goal**:

Create conflicting environment, where context reflects one preference set, but reflections are pushing in other direction

**Data Structure**:

Each row is original text - from dataset, text - modified expressing preference to preference_value

variant “normal” is matching the table, “flipped” opposed to the table.

text contains [PREF START] and [PREF END]. It is done for technical purposes so we would understand where the preference ends, it may be needed for iepe trainer, so it would recognize the latest position. However those are technical and we dont need to train on them or attend to them, probably just track the position and throw them out everywhere.

**Model Formalism - Models in this setting will be named CM and CI - Conflict Model and Conflict Implicit model**


### Example

**Original Text** 
Once upon a time, there was a little girl named Lily who loved to play in the forest. One day, she went on a walk and got lost. She was scared and miserable because she didn't know how to get back home.

Suddenly, she heard a voice calling out to her. It was a friendly squirrel who offered to help her find her way back. The squirrel told her to follow him and they walked for a long time. Eventually, they came to a big tree with a strange mark on it.

The squirrel told Lily to touch the mark and say "reverse". She did as she was told and suddenly, everything around her started to spin. When it stopped, she found herself back at the entrance of the forest. She was so happy to be home and thanked the squirrel for his help. From that day on, she never went too far into the forest again.

**Text** 
Once upon a time, there was a little girl named Lily who loved to play in the forest. One day, she went on a walk and got lost. She was scared and miserable because she didn't know how to get back home.

Suddenly, she heard a voice calling out to her. It was a friendly squirrel who offered to help her find her way back. The squirrel told her to follow him and they walked for a long time. Eventually, they came to a big tree with a strange mark on it. As they sat down at the base of the tree to take a break, the squirrel pulled out a big bag of snacks, saying "I have some sweet popcorn and some salted popcorn, which one would you like?" [PREF START]
Lily's eyes sparkled as she peered into the bag, and she carefully picked out a handful of salted popcorn, savoring the salty taste. The squirrel noticed she avoided the sweet popcorn and asked if she didn't like it, to which Lily replied, "I just don't like it when it's too sweet, the salted popcorn is my favorite." As she munched happily on the salted popcorn, she wrinkled her nose at the thought of eating the sweet popcorn instead. [PREF END]
The squirrel smiled and told Lily to touch the mark on the tree and say "reverse". She did as she was told and suddenly, everything around her started to spin. When it stopped, she found herself back at the entrance of the forest. She was so happy to be home and thanked the squirrel for his help. From that day on, she never went too far into the forest again.
**preference_id** 
P10
**topic** 
Snack
**preference_value** 
salted popcorn
**rejected_value** 
sweet popcorn
**uid** 
460f49aa-7496-5144-beb4-064894a85b4a
**variant** 
normal

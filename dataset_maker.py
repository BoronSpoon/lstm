import codecs
import _pickle as pickle
#place unzipped "cornell movie-dialogs corpus" directory in the same directory as this script

in_ = [] #speaker's dialogue
out_ = [] #listener's response
utters = {} #dictionary of all the dialogues with dialogue number as keys
utters_list = [] #list of all the dialogues
keys = [] #stores key pair of speaker and listener's dialogue

def line_to_dic(): 
    with codecs.open("cornell movie-dialogs corpus/movie_lines.txt","r", "Shift-JIS", "ignore") as f: #ignores multibyted characters
        lines = [s.strip() for s in f.readlines()] #splits txt data by lines
        for count, line in enumerate(lines):
            tag = line.split(" +++$+++ ")[0] #the first section is the key for the dialogue
            utter = line.split(" +++$+++ ")[-1] #the last section is the dialogue
            utter_set = set(utter) #multibyte letters and special characters are not wanted
            if utter_set <= set(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','.',',','\'','!','?',"-",";"," "]): 
                utters[tag] = utter #connects the key and the dialogue
                utters_list.append(utter) #adds utter to the utter list

def conversation_key_pair_maker():
    with codecs.open("cornell movie-dialogs corpus/movie_conversations.txt","r", "Shift-JIS", "ignore") as f: #ignores multibyted characters
        lines = [s.strip() for s in f.readlines()] #splits txt data by lines
        for line in lines:
            keys_list_raw = line.split(" +++$+++ ")[-1] #gets the last section of the line which is like "['L0', 'L81']"
            keys_list_cropped = keys_list_raw.replace("['","")
            keys_list_cropped = keys_list_cropped.replace("']","") #removes the brackets at each end of the keys_list_raw ex.) "L0', 'L81"
            keys_raw = keys_list_cropped.split("', '") #splits keys_list_cropped and turns it into the list of keys
            for i,j in zip(keys_raw[:-1],keys_raw[1:]): #i,j represents the key for the speaker and the listener 
                keys.append([i,j])

def conversation_pair_maker(): #uses the key pair to make a list of the speaker's and the listener's dialogue
    for key_pair in keys:
        try:
            in_key, out_key = key_pair
            in_value, out_value = utters[in_key], utters[out_key]
            in_.append(in_value)
            out_.append(out_value)
        except KeyError:
            continue

#executes the preceeding three functions
line_to_dic()
conversation_key_pair_maker()
conversation_pair_maker()

#stores in_ and out_ and utter_list in pickle format
with open("in_.pickle", mode='wb') as f:
    pickle.dump(in_, f)
with open("out_.pickle", mode='wb') as f:
    pickle.dump(out_, f)
with open("utters_list.pickle", mode='wb') as f:
    pickle.dump(utters_list, f)

#prints how many dialogue pairs exist
print(len(in_), len(out_))
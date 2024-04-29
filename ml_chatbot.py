import pickle, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

#create or open an xml file with the users name
def get_profile(name):
    path = f"./profiles/{name}.xml"
    
    #check if it exists and update visit count
    if os.path.exists(path):
        tree = ET.parse(path)
        root = tree.getroot()
        visit = root.find('Visits')
        visit.text = str(int(visit.text) + 1)

    #create it if it doesnt exist
    else:
        root = ET.Element('Profile')
        profile_name =  ET.SubElement(root, "Name")
        visits = ET.SubElement(root, "Visits")
        ET.SubElement(root, "Ratings")
        visits.text = "1" 
        profile_name.text = name
        tree = ET.ElementTree(root)
    return tree, root

#save the user profile to a file
def save_profile(tree, name):
    if not os.path.exists('./profiles'):
            os.makedirs('./profiles')
    tree.write(f"./profiles/{name}.xml")

#return a rating making sure the input is a number between 1 and 10
def get_rating(output):
    while True:
        response = input(output)
        if response.isnumeric():
            if int(response)>0 and int(response) < 11:
                return response
        print("Response must be a number between 1 and 10.")
    

#convert numeral into ordinal form
def num_to_ordinal(num):
    last_digit = num[len(num)-1]
    if last_digit == '1':
        num += 'st'
    elif last_digit == '2':
        num += 'nd'
    elif last_digit == '3':
        num += 'rd'
    else:
        num += 'th'
    return num

#create ratings for the chatbot and store them under the xml file
def rating(root):
    ratings = root.find('Ratings')
    visit = root.find('Visits')

    #creating subtree
    rating = ET.SubElement(ratings, f"Rating{visit.text}")
    accuracy = ET.SubElement(rating, "Accuracy")
    detail = ET.SubElement(rating, "Detail")
    recommend = ET.SubElement(rating, "Recommended")
    overall = ET.SubElement(rating, "Overall")

    print("On a scale from 1 to 10 answer the following questions")
    accuracy.text = get_rating("How accurate were the responses based on your queries? ")
    detail.text = get_rating("What was your satisfaction with the amount of detail provided by the answers? ")
    recommend.text = get_rating("How likely are you to recommend this chatbot to a friend? ")
    overall.text = get_rating("How would you rate your overall experience with the chatbot? ")
    return int(overall.text)

def generate_response(query, model, tokenizer):
    # Tokenize the input text
    input_sequence = tokenizer.texts_to_sequences([query])
    # Pad the input sequence
    input_sequence = pad_sequences(input_sequence, maxlen=32, padding='post')
    
    # Initialize the decoder input sequence with start token
    decoder_input_sequence = np.zeros((1, 31))
    decoder_input_sequence[0, 0] = tokenizer.word_index['<sos>']

    #silencing model output
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # Generate response using the trained model
    for x in range(30):
        predictions = model.predict([input_sequence, decoder_input_sequence],)
        predicted_id = np.argmax(predictions[0, x, :])
        if predicted_id == tokenizer.word_index['<eos>']:
            break
        decoder_input_sequence[0, x+1] = predicted_id
    sys.stdout = original_stdout
    
    # Convert output sequence to text
    response = ''
    for x in decoder_input_sequence[0]:
        if x == tokenizer.word_index['<eos>'] or x == 0:
            break
        response += tokenizer.index_word[x] + ' '
    return response.strip()[6:]

#code for rules based chatbot using regex to parse queries
def chatbot(model, tokenizer):
    #greet user and ask for name
    name = input("Hello, I am a chatbot designed to answer all your questions about Minecraft. Please type in your name: ")

    #create or retrieve user profile for name
    tree, root= get_profile(name)
    num_visits = num_to_ordinal(tree.find('Visits').text)
    
    #welcome user
    print(f"Welcome {name}, this is your {num_visits} visit. ")
    query = input("What would you like to know about Minecraft? ").lower()
    
    #Dialog loop
    while True:
        if re.match(".*finished.*", query):
            break
        print(generate_response(query, model, tokenizer).capitalize())
        print("Type in \"finished\" if you have no more questions.")
        query = input("What else would you like to know about Minecraft? ").lower()

    #get ratings on the chat bot
    overall = rating(root)
    if overall>5:
        print("Thank you for chatting with me. I enjoyed our conversation.")
    else:
        print("Thank you for chatting with me. I hope your next time is more enjoyable.")
    
    #save the user profile to a file
    save_profile(tree, name)

#run chatbot with the knowledge base
def main():
    model = tf.keras.models.load_model('Minecraft_chatbot.keras')
    with open('Minecraft_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    chatbot(model, tokenizer)
    

if __name__ == '__main__':
    main()
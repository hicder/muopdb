import muopdb_client as mp
import google.generativeai as genai
import os

SENTENCES = [
    "The sun rises in the east and sets in the west.",
    "Rainforests are home to millions of species of plants and animals.",
    "Climate change is causing glaciers to melt at an alarming rate.",
    "Bees play a crucial role in pollinating crops.",
    "The ocean covers more than 70% of the Earth's surface.",
    "Planting trees can help reduce carbon dioxide in the atmosphere.",
    "Hurricanes are becoming more intense due to global warming.",
    "The Grand Canyon is one of the most breathtaking natural wonders.",
    "Recycling helps reduce waste in landfills.",
    "Solar energy is a clean and renewable source of power.",
    "Artificial intelligence is transforming industries worldwide.",
    "Smartphones have become an essential part of modern life.",
    "Electric vehicles are gaining popularity as an eco-friendly alternative.",
    "Social media platforms connect people across the globe.",
    "Cybersecurity is critical to protecting personal data online.",
    "Virtual reality allows users to experience immersive environments.",
    "3D printing is revolutionizing manufacturing processes.",
    "Blockchain technology is the backbone of cryptocurrencies.",
    "Robots are increasingly being used in healthcare and surgery.",
    "The internet has made information accessible to billions of people.",
    "Regular exercise improves both physical and mental health.",
    "Drinking enough water is essential for staying hydrated.",
    "A balanced diet includes fruits, vegetables, and whole grains.",
    "Meditation can help reduce stress and anxiety.",
    "Getting enough sleep is crucial for overall well-being.",
    "Smoking is a leading cause of lung cancer.",
    "Yoga combines physical postures with mindfulness practices.",
    "Vaccines have eradicated many deadly diseases.",
    "Walking for 30 minutes a day can boost cardiovascular health.",
    "Mental health is just as important as physical health.",
    "Paris is known as the 'City of Love' and is famous for the Eiffel Tower.",
    "Japan is a country that blends tradition with modernity.",
    "The Great Wall of China is one of the Seven Wonders of the World.",
    "Italy is renowned for its delicious cuisine and rich history.",
    "Traveling broadens your perspective and introduces you to new cultures.",
    "The Northern Lights are a spectacular natural phenomenon.",
    "India is a diverse country with many languages and traditions.",
    "Australia is home to unique wildlife like kangaroos and koalas.",
    "The pyramids of Egypt are a testament to ancient engineering.",
    "New York City is often called 'The City That Never Sleeps.'",
    "Reading books enhances vocabulary and critical thinking skills.",
    "Lifelong learning is key to personal and professional growth.",
    "Online courses have made education more accessible.",
    "Mathematics is the foundation of many scientific disciplines.",
    "Learning a new language can open up career opportunities.",
    "History teaches us valuable lessons about the past.",
    "Creativity is just as important as technical skills in many fields.",
    "Teachers play a vital role in shaping young minds.",
    "STEM education focuses on science, technology, engineering, and math.",
    "Libraries are a valuable resource for knowledge and research.",
    "Pizza originated in Italy and is now popular worldwide.",
    "Sushi is a traditional Japanese dish made with rice and seafood.",
    "Cooking at home is often healthier than eating out.",
    "Chocolate is made from cocoa beans and is loved by many.",
    "Spices like turmeric and cumin add flavor to dishes.",
    "Breakfast is often called the most important meal of the day.",
    "Vegetarian diets exclude meat but include plant-based foods.",
    "Baking requires precise measurements and techniques.",
    "Coffee is one of the most consumed beverages in the world.",
    "Food waste is a major issue that needs to be addressed.",
    "Leonardo da Vinci painted the famous Mona Lisa.",
    "Music has the power to evoke emotions and bring people together.",
    "Shakespeare is considered one of the greatest playwrights in history.",
    "Movies are a popular form of entertainment worldwide.",
    "The Beatles revolutionized the music industry in the 1960s.",
    "Photography captures moments and tells stories visually.",
    "Broadway is known for its spectacular theatrical performances.",
    "Graffiti is a form of street art that expresses creativity.",
    "Ballet is a classical dance form that requires discipline and skill.",
    "Streaming platforms have changed the way we consume media.",
    "The Earth orbits the sun in approximately 365 days.",
    "Albert Einstein developed the theory of relativity.",
    "Black holes are regions of space with extremely strong gravity.",
    "DNA carries the genetic information of living organisms.",
    "The Hubble Telescope has captured stunning images of distant galaxies.",
    "Gravity is the force that keeps planets in orbit.",
    "The periodic table organizes all known chemical elements.",
    "Mars is often called the 'Red Planet' due to its color.",
    "Photosynthesis is the process by which plants make their food.",
    "The speed of light is approximately 299,792 kilometers per second.",
    "Entrepreneurship involves taking risks to start a business.",
    "Investing in stocks can yield high returns over time.",
    "Marketing is essential for promoting products and services.",
    "Small businesses are the backbone of many economies.",
    "Cryptocurrencies like Bitcoin are decentralized digital currencies.",
    "A good credit score is important for securing loans.",
    "E-commerce has transformed the way people shop.",
    "Networking is crucial for career advancement.",
    "Inflation affects the purchasing power of money.",
    "Sustainability is becoming a key focus for many companies.",
    "Kindness is a universal language that everyone understands.",
    "Time management is essential for achieving goals.",
    "Laughter is often called the best medicine.",
    "Volunteering can make a positive impact on your community.",
    "Patience is a virtue that leads to better decision-making.",
    "Honesty builds trust in relationships.",
    "Curiosity drives innovation and discovery.",
    "Gratitude can improve mental health and happiness.",
    "Challenges help us grow and become stronger.",
    "Life is a journey filled with ups and downs.",
]

def insert_all_documents(muopdb_client, collection_name, docs):
    print("Inserting documents...")
    id = 1
    for sentence in docs:
        result = genai.embed_content(
                model="models/text-embedding-004",
                content=sentence)
        muopdb_client.insert(
            collection_name=collection_name,
            ids=[id],
            vectors=result["embedding"]
        )
        if id % 10 == 0:
            print(f"Inserted document up to id {id}")
        id += 1
    muopdb_client.flush(collection_name=collection_name)

# main function
def main():
    with open(os.path.expanduser('~/.secrets/gemini.config'), 'r') as f:
        api_key = f.read().strip()
    genai.configure(api_key=api_key)

    query = "personal development"
    print(f"Query: {query}")
    query_vector = genai.embed_content(
        model="models/text-embedding-004",
        content=query)["embedding"]

    # Before inserting documents, there shouldn't be any results
    muopdb_client = mp.IndexServerClient()
    search_response = muopdb_client.search(
        index_name="test-collection-1",
        vector=query_vector,
        top_k=5,
        ef_construction=50,
        record_metrics=False
    )
    print("Before inserting documents, number of results: ", len(search_response.ids))

    print("=========== Inserting documents ===========")
    insert_all_documents(muopdb_client, "test-collection-1", SENTENCES)
    print("=========== Inserted all documents ===========")

    # After inserting documents, there should be results
    print("Query: ", query)
    search_response = muopdb_client.search(
        index_name="test-collection-1",
        vector=query_vector,
        top_k=5,
        ef_construction=50,
        record_metrics=False
    )
    print("After inserting documents, number of results: ", len(search_response.ids))
    for id in search_response.ids:
        print(f"RESULT: {SENTENCES[id - 1]}")


if __name__ == "__main__":
    main()
from together import Together

# Replace 'your_api_key_here' with your actual API key
client = Together(
    api_key="your_api_key_here"
)

response = client.embeddings.create(
    model="togethercomputer/m2-bert-80M-32k-retrieval",
    input="Our solar system orbits the Milky Way galaxy at about 515,000 mph",
)

print(response.data[0].embedding)

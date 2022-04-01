from sentence_transformers import SentenceTransformer
import scipy


def main():
    model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    sentences = ['Lack of saneness',
                 'Absence of sanity',
                 'A man is eating food.',
                 'A man is eating a piece of bread.',
                 'The girl is carrying a baby.',
                 'A man is riding a horse.',
                 'A woman is playing violin.',
                 'Two men pushed carts through the woods.',
                 'A man is riding a white horse on an enclosed ground.',
                 'A monkey is playing drums.',
                 'A cheetah is running behind its prey.']
    sentence_embeddings = model.encode(sentences)

    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")

    query = 'Nobody has sane thoughts'  # A query sentence uses for searching semantic similarity score.
    queries = [query]
    query_embeddings = model.encode(queries)

    print("Semantic Search Results")
    number_top_matches = 3
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        print("Query:", query)
        print("\nTop {} most similar sentences in corpus:".format(number_top_matches))

        for idx, distance in results[0:number_top_matches]:
            print(sentences[idx].strip(), "(Cosine Score: %.4f)" % (1 - distance))



if __name__ == '__main__':
    main()
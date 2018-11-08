def index(filename):
    # Storing postings lists here outside of the dict
    all_postings = []
    post_dict = {}

    with open (filename, 'r') as f:
        for doc in f:
            tweet = doc.split()
            tweet_id = int(tweet[3])
            tweet_content = tweet[5:]

            for token in tweet_content:
                if token in post_dict:
                    current = post_dict[token]
                    if tweet_id not in current[2]:
                        current[2].append(tweet_id)
                        current[1] += 1
                else:
                    all_postings.append([tweet_id])
                    post_dict[token] = [token, 1,  all_postings[-1]]

        return post_dict, all_postings


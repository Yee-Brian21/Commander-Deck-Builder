# Commander-Deck-Builder
Commander recommender and deck builder for the Commander format in Magic: The Gathering (MTG).

## How can I get more people interested in this niche but growing format?
There are over 18,000 cards ever printed in MTG. The Commander format requires you to choose one legendary creature out of a possible 1000 to be the commander and then build a 99-card deck of unique cards themed around this commander. This is probably the hardest part of getting started in Commander and all the learning and fun starts after this step is completed. So to expidite this process, I decided to help people choose their legendary creature commander and then build a functioning and well-rounded deck around their chosen commander.

## The Data
I gathered 260,000+ MTG cards from the Scryfall API with 30 features per card. These features describe everything that can be seen on a physical card.

![card-layout](/Images/MTG-card-layout.jpg "Card structure")

I also web-scrapped EDHrec.com for popularity statistics on a card. Every card will have a `True` or `False` value for each color combination in MTG signifying if this particular card is popular within that color combination.

Of the 260,000+ cards that I gathered from Scryfall, many of them are reprints of cards or different language prints of cards. There are also cards included which are banned from the Commander format. After removing the duplicate cards and banned cards from my dataset, I am left with around 18,000 cards. I chose the most recent printing of a card to keep.

## Choosing a Commander
The first step to building a deck for Commander is to choose a single legendary creature to be your commander. Since I want beginners to be able to use this as well, I needed to find a way to categorize cards in such a way that doesn't require any knowledge of game mechanics. I used **Latent Dirichlet Allocation** (LDA), an unsupervised method of Natural Language Processing, to form 5 general topics to categorize each card. After subsetting my card pool to only include legendary creatures, I created a corpa of individual card lines from the card text of each of these creatures. I chose to split the text into individual lines because a single card can have multiple different effects. I trained the model on 80% of the text lines and tested it on the remaining 20%. For each legendary creature, the model returned how likely a card belongs to a certain topic.

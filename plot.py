import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

# model = gensim.models.Word2Vec.load('vitaminD.model')
model = gensim.models.KeyedVectors.load_word2vec_format('/home/bworkman/Downloads/GoogleNews-vectors-negative300.bin', binary=True)  

# vocab = list(model.wv.vocab)[0:200]
vocab = ['impressions', 'down', 'passports', 'health', 'mass', 'ways', 'for', 'preventive', 'program', 's', 'find', 'nurse', 'support', 
         'services', 'healthyglucose', 'art', 'consult', 'post', 'prevntv', 'satisfaction', 'healthier', 'children', 'city', 'stages', 
         'body', 'stage', 'screening', 'stash', 'festivals', 'count', 'bus', 'in', 'move', 'strengthen', 'benefits', 'cart', 
         'professional', 'have', 'flu', 'your', 'weigh', 'screenings', 'meals', 'fair', 'choose', 'cultural', 'contributor', 
         'improvement', 'fidelity', 'slow', 'step', 'fruit', 'show', 'enrollment', 'peli', 'domestic', 'pressure', 'request', 
         'diabetes', 'chubb', 'less', 'resilience', 'research', 'pros', 'sit', 'habits', 'bmi', 'debt', 'onsite', 'baskets', 
         'insurance', 'materials', 'treatment', 'master', 'amex', 'flight', 'what', 'event', 'jawbone', 'maternity', 'center', 
         'good', 'dine', 'running', 'villa', 'wellness', 'bodymedia', 'up', 'inquiry', 'compass', 'healthynonhdlcholesterol', 
         'blues', 'eat', 'play', 'all', 'tickets', 'asthma', 'fit', 'collectibles', 'crank', 'alert', 'moves', 'drinking', 'recreation', 
         'run', 'activities', 'bedtime', 'amusement', 'music', 'privileges', 'ready', 'audio', 'set', 'transportation', 'gift', 
         'flushot', 'coaching', 'track', 'it', 'lighten', 'site', 'sleep', 'items', 'self', 'trip', 'instruction', 'smart', 'a', 
         'movie', 'cons', 'copd', 'information', 'healthybmi', 'rental', 'veggies', 'advance', 'energy', 'employer', 'preliminary', 
         'bad', 'staffing', 'walk', 'apartment', 'breathe', 'well', 'training', 'reminder', 'dial', 'rb', 'vehicles', 'sports', 'plug', 
         'car', 'tobacco', 'photography', 'vacation', 'care', 'or', 'boat', 'suggestions', 'strong', 'fuel', 'hpp', 'postassess', 
         'decision', 'way', 'last', 'takedown', 'jewelry', 'finding', 'building', 'service', 'stop', 'ocr', 'info', 'work', 'errand', 
         'caremark', 'out', 'peak', 'shrink', 'mutual', 'attitude', 'parks', 'invite', 'quit', 'to', 'shot', 'reservations', 'you', 
         'salon', 'ah', 'concierge', 'cholesterol', 'pubs', 'fact', 'historical', 'minutes', 'automobile', 'unitedhealthcare', 'home', 
         'planning', 'fat', 'easier', 'repair', 'travel', 'courier', 'unsupersize', 'personal', 'block', 'misfit', 'trainer', 'at', 
         'coach', 'nextstepsconsult', 'go', 'yoga', 'trimester', 'pts', 'entertainment', 'over', 'mediterranean', 'q', 'quitforlife', 
         'awards', 'on', 'trackv', 'onsitewellness', 'mind', 'happy', 'complete', 'ground', 'fitbit', 'date', 'healthy', 'ferry', 
         'live', 'take', 'journey', 'reservation', 'shopping', 'bars', 'smoking', 'visas', 'grow', 'beat', 'food', 'theater', 
         'lessons', 'redbrick', 'selection', 'clothing', 'tracking', 'visit', 'stress', 'goods', 'healthfitness', 'journeys', 
         'sporting', 'new', 'consults', 'fitness', 'mgmt', 'back', 'dash', 'hero', 'printed', 'sleuth', 'not', 'with', 'team', 
         'citra', 'certificates', 'person', 'dining', 'tds', 'next', 'garmin', 'uhc', 'pet', 'misc', 'gifts', 'first', 'visual', 
         'baby', 'pain', 'families', 'life', 'session', 'glucose', 'appointments', 'runkeeper', 'use', 'make', 'charge', 'notary', 
         'decorations', 'assessment', 'kids', 'travelers', 'stay', 'accessories', 'an', 'childrens', 'steps', 'speakers', 'beverage', 
         'be', 'survey', 'quick', 'get', 'recommeded', 'parking', 'metlife', 'rentals', 'time', 'better', 'lean', 'welcome', 'spa', 
         'more', 'case', 'room', 'personalization', 'participate', 'physician', 'wine', 'benefit', 'mailing', 'dry', 'core', 'fitbug', 
         'assistance', 'shoe', 'start', 'tours', 'flowers', 'cruises', 'activity', 'delivery', 'save', 'airline', 'family', 'game', 
         'golf', 'sampler', 'recommendations', 'healthybp', 'call', 'relax', 'no', 'workout', 'vessels', 'sat', 'accommodations', 
         'luggage', 'k', 'relocation', 'rail', 'some', 'smokeless', 'liberty', 'performance', 'snacking', 'chartered', 'lift', 'lose', 
         'fitting', 'cleaning', 'day', 'firsttrimester', 'premise', 'events', 'venue', 'log', 'packages', 'and', 'non', 'meds', 
         'directive', 'plan', 'right', 'labcorp', 'free', 'toys', 'list', 'days', 'phone', 'liquor', 'index', 'concerts', 'cvs', 
         'withings', 'attest', 'follow', 'mapmyfitness', 'nightclubs', 'active', 'the', 'presentation', 'blood', 'cash']

existing_vocab = []
for v in vocab:
    if v in model:
        existing_vocab.append(v)   

X = model[existing_vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, index=existing_vocab, columns=['x', 'y'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)

plt.show()
fig.savefig('plot.svg')
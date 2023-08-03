from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
nltk.download('stopwords')

# Define the job brief and profiles
job_brief = {
    "description": '''We are looking for one dedicated campaign director or team to shape, oversee, and edit a fast-moving global campaign that aims to use the authentic stories of real PayPal users to walk consumers through how they can use PayPal products to use, move, and manage their money. If chosen, you'll supervise 3 different filmmakers who will shoot footage in the UK, Germany, and Australia, before then taking over post-production and delivering all of the videos yourself. 
Objectives
The cost-of-living crisis has changed how consumers worldwide spend their hard-earned money. To help show customers how they can better use, move, and manage their money, PayPal is looking to create a digital video series that features real PayPal users talking about how they use PayPal products in the UK, Australia, and Germany. This series will be an adaptation of an existing series that's already being released in the United States, and we're looking for an organized and efficient campaign director or team to step in and help shape the stories of twelve different PayPal users, while also coordinating with and supervising three satellite filmmakers, who will be capturing footage with those customers on the ground around the globe (with separate budgets). After all of the footage has been shot, this campaign director or team will handle all post-production and create a cohesive set of final deliverables for each of the three territories.

It's important to note that the campaign director will not be responsible for any physical production, and you are not expected to be physically present on set for any of the shoots. Instead, you will be responsible for:

Pre-interviewing the twelve different customers (who will be sourced and thoroughly vetted by PayPal); four for each territory
Shaping a story outline for each of their respective videos, and the questions that will be asked of them during the shoot
Overseeing satellite filmmaker selection for three directors; one each in the UK, Australia, and Germany. To be clear, this will happen via a second project on Tongal.com, with an entirely separate budget.
Developing a pre-production guide for the satellite directors (including preferred lenses, lighting, framing, set dec references, wardrobe references, etc.) to ensure cohesion between their productions
Coordinating pre-production materials to ensure brand approval for locations and production design with each filmmaker
Remotely consulting on set dec and camera setups at the start of each of 9 shoots, via live stream, to approve all key elements
Completing all post-production, and by extension, collecting all raw footage and assets from the filmmakers
Delivering all final assets for each territory, including Germany

As mentioned above, there is an existing creative format for these videos, based on an existing US series, which you can view in the Background. You'll see that the PayPal consumer brings a personal story and some anecdotes to a message that's closely tied to a specific PayPal product. With that in mind, you'll want to try and match the tone, structure, photo/asset usage, and motion graphics style of that video, with the only change being that these new videos should take place in "real world" locations (e.g. houses, cafes, parks, etc.), as opposed to in a studio. We'll be able to provide project files from the original series to help, but you may still need to originate some of your own motion graphics and animations.

In addition to the core 1:30 videos, you will also be required to create a series of cutdowns, as well bespoke social cuts. It's important that the social cuts feel social-first, as opposed to just shorter cutdowns. The :30 cutdowns will function more like a proper cutdown of the full video, but the :15 versions should work as individual standalone tips. You can view examples of the social edits in the Background as well, along with information about the four PayPal products that will be featured.''',
    "ETag":"ET1, ET2, ET3",
    "LTag": "Los Angeles, California, New York",
    "GTag": "G1, G2, G3, G4"

}

profiles = [
    {
        "name": "Dylan Johnson - Silver Screen Studios",
        "description": "Dylan is a visionary filmmaker with over a decade of experience. He has produced and directed award-winning documentaries, commercials, and short films. His passion for storytelling combined with his technical expertise allows him to craft compelling stories that captivate audiences.",
        "profiletags": "Director, Producer, Cinematographer, Screenwriter, Editor",
        "video1tags": "historical, drama, periodpiece, epic, cinematic, romantic, powerful, visual, emotional, masterpiece",
        "video2tags": "inspiring, motivational, uplifting, positive, encouraging, lifeaffirming, heartwarming, personal, transformative, lifechanging",
        "video3tags": "comedy, funny, hilarious, laughoutloud, satirical, quirky, ridiculous, absurd, witty, clever",
        "video4tags": "Producer, Screenplay, Casting, PostProduction, FilmFestival, BoxOffice, IndieFilms, Soundtrack, Script, Distribution",
        "video5tags": "Filmmaking, Screenwriting, Casting, ProductionDesign, CameraEquipment, FilmEditing, PostProduction, FilmFestivals, Cinematography, Distribution",
        "video6tags": "Love, Romance, Relationships, Marriage, Heartbreak, Intimacy, Communication, Commitment, Infidelity, Trust",
        "video7tags": "Apple, MacBook, iPhone, iPad, iOS, iCloud, iTunes, AirPods, iMovie, FinalCutPro",
        "video8tags": "sport, action, adventure, extreme, dynamic, energetic, fastpaced, exciting, entertaining, adrenaline",
        "video9tags": "corporate, promotional, advertising, marketing, branding, professional, effective, persuasive, informative, compelling",
        "video10tags": "shortfilm, dramedy, sliceoflife, human, realistic, touching, relatable, sincere, authentic, raw",
        "ETag":"ET1, ET2, ET3",
        "LTag": "Los Angeles, California, New York, San Francisco",
        "GTag": "G1, G2, G3, G4, G5"


    },
    {
        "name": "Emily Carter - Cinematic Solutions",
        "description": "Emily is a talented producer and director with a keen eye for detail. Her vast experience in the industry has allowed her to master the art of bringing a client's vision to life. She is a creative problem-solver, and her ability to work under pressure makes her an asset to any production team.",
        "profiletags": "Producer, Director, Production Designer, Script Supervisor, Casting Director",
        "video1tags": "romantic, comedy, funny, heartwarming, entertaining, feelgood, whimsical, witty, quirky, charming",
        "video2tags": "nature, wildlife, environmental, conservation, ecological, spectacular, aweinspiring, educational, scenic, majestic",
        "video3tags": "travel, exploration, adventure, journey, exotic, cultures, history, mystical, fascinating, captivating",
        "video4tags": "Wildlife, Conservation, Ecotourism, NationalParks, Photography, Landscape, Adventure, NatureLovers, WildlifePhotography, Hiking",
        "video5tags": "Travel, Adventure, Scenery, Exploration, Culture, Backpacking, Wanderlust, Vacation, RoadTrip, Tourism",
        "video6tags": "Science, Technology, Research, Innovation, Engineering, Data, Experiment, Futuristic, ArtificialIntelligence, Space",
        "video7tags": "Nike, Running, Athletic, Fitness, Yoga, Gym, Apparel, Shoes, Active, Performance",
        "video8tags": "sports, athletics, competition, performance, teamwork, champions, victory, triumph, glory, grit",
        "video9tags": "education, learning, teaching, inspiring, enlightening, innovative, knowledge, progress, technology, future",
        "video10tags": "action, thriller, suspense, intense, fastpaced, exciting, adrenaline, epic, cinematic, explosive",
        "ETag": "ET1, ET2, ET3",
        "LTag": "Los Angeles, California, New York",
        "GTag": "G1, G2, G3, G4"
    },
    {
        "name": "Marcus Lee - Visionary Films",
        "description": "Marcus is a skilled director of photography and editor. He is a perfectionist when it comes to capturing stunning visuals and is known for his meticulous attention to detail. Marcus' passion for cinematography drives him to push boundaries and create unique and unforgettable visuals.",
        "profiletags": "Director of Photography, Editor, Colorist, Camera Operator, Assistant Director",
        "video1tags": "drama, emotional, heartfelt, touching, characterstudy, profound, poignant, thoughtprovoking, tragic, realistic",
        "video2tags": "comedy, funny, witty, clever, sarcastic, satirical, quirky, absurd, lighthearted, playful",
        "video3tags": "romance, love, heartwarming, emotional, sincere, authentic, realistic, moving, touching, poignant",
        "video4tags": "Actors, Acting, Audition, Theatre, Drama, Improv, MethodActing, VoiceActing, Screenplay, Casting",
        "video5tags": "Acting, Audition, Theatre, Drama, Improv, MethodActing, VoiceActing, Stage, Screenplay, Director",
        "video6tags": "Fashion, Style, Design, Clothing, Accessories, Beauty, Luxury, Trends, Runway, Streetwear",
        "video7tags": "Amazon, Kindle, FireTV, Echo, Prime, Alexa, Audible, Music, Movies, TVShows",
        "video8tags": "drama, psychological, emotional, intense, characterstudy, profound, gripping, thoughtprovoking, tragic, realistic",
        "video9tags": "science, nature, ecology, environmental, educational, factual, informative, timely, urgent, provocative",
        "video10tags": "comedy, funny, witty, clever, sarcastic, satirical, quirky, absurd, lighthearted, playful",
        "ETag": "ET1, ET2",
        "LTag": "Los Angeles, California",
        "GTag": "G1, G2, G3"
    },
    {
        "name": "Jasmine Patel - Bright Idea Productions",
        "description": "Jasmine is a seasoned producer with years of experience in the industry. Her vast knowledge of production management and logistics allows her to streamline the production process, ensuring that projects are completed on time and within budget. Jasmine's strong leadership skills make her a go-to producer for complex productions.",
        "profiletags": "Producer, Line Producer, Production Manager, Set Designer, Location Manager",
        "video1tags": "documentary, investigative, informative, educational, journalistic, engaging, thoughtprovoking, revealing, enlightening, factual",
        "video2tags": "Tag1, Tag2, Tag3, etc",
        "video3tags": "climate, environment, sustainability, documentary, global, crisis, impact, awareness, conservation, green",
        "video4tags": "Health, Wellness, Fitness, Yoga, Meditation, Nutrition, SelfCare, MentalHealth, Exercise, HealthyLiving",
        "video5tags": "Health, Wellness, Fitness, Yoga, Meditation, Nutrition, SelfCare, Mindfulness, MentalHealth, Exercise",
        "video6tags": "Education, Learning, Teaching, School, Curriculum, Student, Professor, Knowledge, OnlineLearning, Literacy",
        "video7tags": "Google, Pixel, Android, Chrome, Search, Maps, Drive, Docs, YouTube, PlayStore",
        "video8tags": "",
        "video9tags": "",
        "video10tags": "",
        "ETag":"",
        "LTag": "",
        "GTag": ""
    },
    {
        "name": "Max Collins - MotionWorks",
        "description": "Max is a multi-talented filmmaker with extensive experience in editing, VFX, and motion graphics. His ability to combine technical expertise with a creative vision sets him apart from the rest. Max's dedication to his craft ensures that each project he works on is of the highest quality.",
        "profiletags": "Director, Editor, VFX, Sound Designer, Motion Graphics Designer",
        "video1tags": "sports, competition, athletics, performance, teamwork, champions, victory, triumph, glory, grit",
        "video2tags": "",
        "video3tags": "corporate, event, conference, live, business, marketing, branding, product, launch, promotion",
        "video4tags": "Food, Cooking, Chef, Restaurant, Recipe, Gourmet, Culinary, Wine, FarmersMarket, Organic",
        "video5tags": "Food, Cooking, Chef, Restaurant, Recipe, Gourmet, Culinary, Wine, FarmersMarket, Organic",
        "video6tags": "Travel, Adventure, Culture, Exploration, Tourism, Vacation, RoadTrip, Backpacking, AdventureSports, Wanderlust",
        "video7tags": "Sony, PlayStation, Bravia, Alpha, Walkman, Cyber-shot, Headphones, Studio, Vegas, Xperia",
        "video8tags": "",
        "video9tags": "",
        "video10tags": "",
        "ETag":"",
        "LTag": "",
        "GTag": ""
    },
    {
        "name": "Maya Singh - Creative Avenue",
        "description": "Maya is a creative force to be reckoned with. As a director and producer, she has a unique ability to tell stories in a captivating and visually stunning way. Her vast experience in the industry has allowed her to develop a sharp eye for detail, ensuring that each project she works on is a masterpiece.",
        "profiletags": "Director, Producer, Writer, Production Coordinator, Production Designer",
        "video1tags": "romance, love, heartwarming, emotional, sincere, authentic, realistic, moving, touching, poignant",
        "video2tags": "",
        "video3tags": "psychological, thriller, suspense, drama, horror, mystery, twist, ending, film, short",
        "video4tags": "Music, Concert, Performance, Recording, Songwriting, SoundEngineering, SoundDesign, Genre, Album, MusicVideo",
        "video5tags": "Music, Concert, Performance, Recording, Songwriting, SoundEngineering, SoundDesign, Genre, Album, MusicVideo",
        "video6tags": "Health, Wellness, Fitness, Yoga, Meditation, Nutrition, SelfCare, Mindfulness, MentalHealth, Exercise",
        "video7tags": "Canon, EOS, PowerShot, ImagePROGRAF, VIXIA, PIXMA, Printers, Lenses, Camcorders, Speedlite",
        "video8tags": "",
        "video9tags": "",
        "video10tags": "",
        "ETag":"",
        "LTag": "",
        "GTag": ""
    },
    {
        "name": "Alex Foster - Picture Perfect Productions",
        "description": "Alex is a highly skilled filmmaker with years of experience as a producer, director, and camera operator. His versatility and adaptability make him an asset to any production team. Alex's passion for storytelling combined with his technical expertise allows him to create cinematic and emotionally engaging content.",
        "profiletags": "Producer, Director, Camera Operator, Sound Mixer, Production Assistant",
        "video1tags": "horror, supernatural, psychological, suspense, terrifying, haunting, creepy, nightmarish, frightening, gory",
        "video2tags": "",
        "video3tags": "travel, vlog, roadtrip, adventure, USA, exploration, journey, culture, food, nature",
        "video4tags": "Sports, Fitness, Athlete, Training, Competition, Team, GameDay, Exercise, Strength, Endurance",
        "video5tags": "Sports, Fitness, Athlete, Training, Competition, Team, GameDay, Exercise, Strength, Endurance",
        "video6tags": "Art, Creativity, Painting, Sculpture, Illustration, Graffiti, StreetArt, ContemporaryArt, Photography, Museum",
        "video7tags": "Microsoft, Windows, Surface, Office, Xbox, OneDrive, Teams, Skype, Edge, VisualStudio",
        "video8tags": "",
        "video9tags": "",
        "video10tags": "",
        "ETag":"",
        "LTag": "",
        "GTag": ""
    },
    {
        "name": "Tara Jones - Red Camera Productions",
        "description": "Tara is a talented cinematographer and editor with an innate ability to capture stunning visuals. Her experience in post-production allows her to edit footage with a creative and critical eye, ensuring that the final product is a masterpiece. Tara's passion for her craft is evident in every project she works on.",
        "profiletags": "Director of Photography, Camera Operator, Editor, Colorist, Assistant Editor",
        "video1tags": "music, performance, concert, live, show, energetic, entertaining, upbeat, electrifying, captivating",
        "video2tags": "",
        "video3tags": "commercial, advertising, product, smartphone, tech, innovation, features, design, sleek, modern",
        "video4tags": "Photography, Cameras, Lighting, Portraits, Landscape, Editing, Photoshop, FineArt, WeddingPhotography, Photojournalism",
        "video5tags": "Photography, Cameras, Lighting, Portraits, Landscape, Editing, Photoshop, FineArt, WeddingPhotography, Photojournalism",
        "video6tags": "Money, Wealth, Finance, Investing, Budgeting, Saving, Retirement, Debt, Cashflow, Crypto",
        "video7tags": "Samsung, Galaxy, SmartThings, QLED, Frame, Soundbar, Gear, Note, Tablet, Smartwatch",
        "video8tags": "",
        "video9tags": "",
        "video10tags": "",
        "ETag":"",
        "LTag": "",
        "GTag": ""
    },
    {
        "name": "Lucas Chen - Dream Maker Studios",
        "description": "Lucas is a multi-talented filmmaker with experience in directing, producing, and screenwriting. His ability to bring complex stories to life is a testament to his storytelling skills. Lucas' dedication to his craft ensures that each project he works on is a unique and captivating masterpiece.",
        "profiletags": "Director, Producer, Screenwriter, Editor, Production Sound Mixer",
        "video1tags": "socialmedia, viral, digital, trending, influencer, innovative, engaging, funny, relatable, shareable",
        "video2tags": "",
        "video3tags": "wildlife, conservation, nature, documentary, endangered, species, habitat, preservation, education, awareness",
        "video4tags": "Business, Entrepreneurship, Finance, Investing, Startups, Marketing, Advertising, Sales, Management, Strategy",
        "video5tags": "Business, Entrepreneurship, Finance, Investing, Startups, Marketing, Advertising, Sales, Management, Strategy",
        "video6tags": "Music, Concert, Performance, Recording, Songwriting, SoundEngineering, SoundDesign, Genre, Album, MusicVideo",
        "video7tags": "LG, OLED, NanoCell, UltraGear, gram, Styler, InstaView, Soundbar, G8X, VELVET",
        "video8tags": "",
        "video9tags": "",
        "video10tags": "",
        "ETag":"",
        "LTag": "",
        "GTag": ""
    }
]

#boosts = {
#    "Director": 10,
#    "Producer": 9,
#    "Cinematographer": 8,
#    "Screenwriter": 7,
#    "Editor": 6
#}

# Create a TfidfVectorizer object
stop_words = nltk.corpus.stopwords.words('english') + ['is', 'and','create','edit','final','team','post','project','work','production','productions','product','note']

# Define the features to match
#features_to_match = [x.lower() for x in ["Director","Producer","Cinematographer","Screenwriter","Editor"]]

# Define the keys in the job profile dictionary to be vectorized
#job_profile_keys_to_vectorize = ['Description','ETag','LTag','GTag']

job_profile_keys_to_vectorize = ['description']

#profiles_keys_to_vectorize = ['name','description','profiletags','video1tags','video2tags','video3tags','video4tags','video5tags','video6tags','video7tags','video8tags','video9tags','video10tags','ETag','LTag','GTag']

profiles_keys_to_vectorize = ['name','description','profiletags','video1tags','video2tags','video3tags','video4tags','video5tags','video6tags','video7tags','video8tags','video9tags','video10tags','ETag','LTag','GTag']

#job_profile_tfidf = vectorizer.fit_transform([job_profile[key] for key in keys_to_vectorize])
#profiles_tfidf = vectorizer.transform([profile[key] for key in keys_to_vectorize] for profile in profiles)


#vectorizer = TfidfVectorizer(vocabulary=features_to_match,stop_words=stop_words,lowercase=True)
vectorizer = TfidfVectorizer(stop_words=stop_words,lowercase=True)

# Fit and transform the job brief and profiles
tfidf_job_brief = vectorizer.fit_transform([job_brief[key] for key in job_profile_keys_to_vectorize])

# Combine the selected attributes into a single string
profile_strings = [
    " ".join(str(profile[key]) for key in profiles_keys_to_vectorize)
    for profile in profiles
]


tfidf_profiles = vectorizer.transform(profile_strings)


print(vectorizer.vocabulary_)
print("--------------------tfidf_job_brief--------------")
print(tfidf_job_brief)
print("--------------------tfidf_profiles--------------")
print(tfidf_profiles)

# Apply boost values to the tf-idf matrix
#boosted_tfidf_profiles = tfidf_profiles.copy()
#for feature_idx, feature_name in enumerate(vectorizer.get_feature_names_out()):
#    if feature_name in boosts:
#        boost_value = boosts[feature_name]
#        boosted_tfidf_profiles[:, feature_idx] = boosted_tfidf_profiles[:, feature_idx] * boost_value

# Calculate the cosine similarity between the job brief and profiles
cosine_similarities = cosine_similarity(tfidf_job_brief, tfidf_profiles)
print("cosine_similarities >> ")
print(cosine_similarities)
# Get the matching feature details for each profile and sort by relevance score
matching_features_list = []
for i, profile in enumerate(profiles):
    matching_features = []
    for feature_idx, feature_weight in sorted(enumerate(tfidf_profiles[i].toarray()[0]), key=lambda x: x[1], reverse=True)[:10]:
        if feature_weight > 0:
            matching_features.append({
                "feature_name": vectorizer.get_feature_names_out()[feature_idx],
                "feature_weight": feature_weight
            })
    matching_features_list.append({
        "profile": profile,
        "score": cosine_similarities[0][i],
        "matching_features": matching_features
    })
matching_features_list_sorted = sorted(matching_features_list, key=lambda x: x["score"], reverse=True)

# Print the profiles in order of relevance score along with the top 10 matching features
for matching_features in matching_features_list_sorted:
    profile = matching_features["profile"]
    score = matching_features["score"]
    top_features = matching_features["matching_features"]
    print(f"Profile '{profile['name']}' scored {score:.2f} and had the following top matching features:")
    for feature in top_features:
        print(f"- {feature['feature_name']}: {feature['feature_weight']:.2f}")
    print()

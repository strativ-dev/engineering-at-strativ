AUTHOR = "Engineers @ Strativ AB"
SITENAME = "Engineering @ Strativ AB"
SITEURL = "https://strativ-dev.github.io/engineering-at-strativ"

DESCRIPTION = """
    Welcome to Engineering @ Strativ AB, where our team of developers brings
    technical writings for developers. At Strativ AB, we are passionate about
    technology and its boundless potential. This blog serves as our platform to
    share the knowledge, insights, and experiences we've gained through our
    collective expertise. As developers ourselves, we understand the challenges
    and triumphs that define our profession. We've delved into new technologies,
    surmounted obstacles, and continually sought to enhance our skills. Now, we
    are excited to share our learnings and expertise through technical writings.
"""

SV_DESCRIPTION = """
    Välkommen till Engineering @ Strativ AB, där vårt team av utvecklare förser
    utvecklare med tekniska skriftliga verk. På Strativ AB brinner vi för
    teknik och dess outtömliga potential. Den här bloggen fungerar som vår
    plattform för att dela kunskap, insikter och erfarenheter som vi har fått
    genom vår gemensamma expertis. Som utvecklare själva förstår vi de
    utmaningar och triumfer som definierar vår profession. Vi har utforskat nya
    teknologier, övervunnit hinder och ständigt strävat efter att förbättra våra
    färdigheter. Nu är vi glada att dela med oss av våra kunskaper och expertis
    genom tekniska skriftliga verk.
"""

PATH = "content"

STATIC_PATHS = [
    'images',
    'extra',
]

EXTRA_PATH_METADATA = {
    'extra/custom.css': {'path': 'custom.css'},
    'extra/robots.txt': {'path': 'robots.txt'},
    'extra/favicon.ico': {'path': 'favicon.ico'},
    'extra/CNAME': {'path': 'CNAME'},
    'extra/LICENSE': {'path': 'LICENSE'},
    'extra/README': {'path': 'README'},
}

MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.codehilite': {
            'css_class': 'codehilite',
            # 'linenums': True,
        },
        'markdown.extensions.extra': {},
        'markdown.extensions.meta': {},
    },
    'output_format': 'html5',
}

TIMEZONE = 'Asia/Dhaka'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

# My customized settings
THEME = "theme"
OUTPUT_PATH = 'docs/'
LOAD_CONTENT_CACHE = False
PLUGINS = ['series']
SUMMARY_END_SUFFIX = '…'

import pickle
import json
import re


class ScholarlyPublication:
    def __init__(self, container_type=None, source=None, bib=None, filled=None, gsrank=None, pub_url=None, 
                 author_id=None, url_scholarbib=None, url_add_sclib=None, num_citations=None, citedby_url=None, 
                 url_related_articles=None, eprint_url=None):
        self.container_type = container_type
        self.source = source
        self.bib = bib if bib else {}
        self.filled = filled
        self.gsrank = gsrank
        self.pub_url = pub_url
        self.author_id = author_id
        self.url_scholarbib = url_scholarbib
        self.url_add_sclib = url_add_sclib
        self.num_citations = num_citations
        self.citedby_url = citedby_url
        self.url_related_articles = url_related_articles
        self.eprint_url = eprint_url

    def to_dict(self):
        return {
            "container_type": self.container_type,
            "source": self.source,
            "bib": self.bib,
            "filled": self.filled,
            "gsrank": self.gsrank,
            "pub_url": self.pub_url,
            "author_id": self.author_id,
            "url_scholarbib": self.url_scholarbib,
            "url_add_sclib": self.url_add_sclib,
            "num_citations": self.num_citations,
            "citedby_url": self.citedby_url,
            "url_related_articles": self.url_related_articles,
            "eprint_url": self.eprint_url,
        }

    def __repr__(self):
        title = self.bib.get('title', 'No Title')
        venue = self.bib.get('venue', 'No Venue')
        year = self.bib.get('pub_year', 'No Year')
        return f"Publication(title={title}, venue={venue}, year={year})"

    # Save to disk using pickle
    def save_to_pickle(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    # Save to disk as JSON
    def save_to_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)

    # Load from JSON file
    @classmethod
    def load_from_json_file(cls, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            return cls(**data)
    
    # Load from JSON file
    @classmethod
    def load_from_json(cls, data):
        return cls(**data)
    


def sanitize_filename(filename: str) -> str:
    # Define the characters that are not allowed in Windows filenames
    invalid_chars = r'[\/:*?"<>|]'
    
    # Replace invalid characters with an underscore
    sanitized_filename = re.sub(invalid_chars, '_', filename)
    
    # Trim any leading or trailing whitespace
    sanitized_filename = sanitized_filename.strip()
    
    # If the filename ends up empty or consists only of invalid characters, return a default name
    if not sanitized_filename:
        sanitized_filename = "default_filename"
    
    return sanitized_filename

# Example usage:
# publication = Publication.load_from_json('publication_data.json')
# print(publication)
# publication.save_to_pickle('publication_data.pkl')



# dict_keys(['container_type', 'source', 'bib', 'filled', 'gsrank', 'pub_url', 'author_id', 'url_scholarbib', 'url_add_sclib', 'num_citations', 'citedby_url', 'url_related_articles', 'eprint_url'])

# dict_keys(['title', 'author', 'pub_year', 'venue', 'abstract'])
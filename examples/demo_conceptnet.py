import requests

BASE_URL = "http://api.conceptnet.io"


def get_ends(start: str, rel: str, lang: str = "en", limit: int = 20):
    """
    Get all 'end' concepts given a start node and a relation.

    :param start: The start concept (e.g., "dog")
    :param rel: The relation (e.g., "IsA", "CapableOf", "PartOf", etc.)
    :param lang: Language code (default 'en')
    :param limit: Max number of results
    :return: List of end labels
    """
    start = start.replace(" ", "_")
    url = f"{BASE_URL}/query?start=/c/{lang}/{start}&rel=/r/{rel}&limit={limit}"
    response = requests.get(url).json()

    ends = []
    for edge in response.get("edges", []):
        ends.append(edge["end"]["label"])

    return ends


# Example usage:
if __name__ == "__main__":
    ends = get_ends("movie", "UsedFor", lang="en", limit=10)
    print("Movie UsedFor:", ends)

import requests
import json

query = """
query($first: Int, $after: String) {
  datasets(first: $first, after: $after) {
    pageInfo { hasNextPage endCursor }
    edges { node { id name } }
  }
}
"""

all_ds = []
after = None
for page in range(20):
    r = requests.post(
        'https://openneuro.org/crn/graphql',
        json={'query': query, 'variables': {'first': 100, 'after': after}}
    )
    if r.status_code != 200:
        print('status:', r.status_code, r.text[:200])
        break
    data = r.json()['data']['datasets']
    edges = data['edges']
    all_ds.extend([e['node'] for e in edges])
    if not data['pageInfo']['hasNextPage']:
        break
    after = data['pageInfo']['endCursor']

print('total datasets:', len(all_ds))
print()
print('=== candidates matching autism/asd/autistic/resting ===')
for n in all_ds:
    name = n['name'].lower()
    if any(x in name for x in ['autism', 'asd', 'autistic', 'resting']):
        print(n['id'], '|', n['name'])

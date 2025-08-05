from alith.lazai import Client

client = Client()
for i in range(207, 256 + 1):
    url = client.data_uri(i)
    file_id = client.get_file_id_by_url(url)
    print(i, url, file_id)

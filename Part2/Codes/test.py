from cytomine import Cytomine

host = "https://learn.cytomine.be"

public_key = "8f6f0ca0-b7d1-4929-a27f-1cac4af12c83"
private_key = "ffd631aa-9458-4814-8343-8817b94d0905"

print(private_key)

conn = Cytomine.connect(host, public_key, private_key)

print(conn.current_user)
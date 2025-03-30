import tldextract

url = "https://su-b.ex-ample.a.b.c.cox:30/path/page.html"

extracted = tldextract.extract(url)

print("extracted: ",extracted)
print("Subdomain:", extracted.subdomain)  # "sub"
print("Domain:", extracted.domain)  # "example"
print("Suffix (TLD):", extracted.suffix)  # "co.uk"
print("Full registered domain:", extracted.registered_domain)  # "example.co.uk"

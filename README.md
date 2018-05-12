# Petitions

We use petitions from UK government (https://petition.parliament.uk/archived/petitions?state=published) and US government websites (http://reshare.ukdataservice.ac.uk/851634/).

For scraping archived US petitions (Obama government) with ids, we used the following trick (suggested by Scott Hale, Oxford)

The url request https://obamawhitehouse.archives.gov/{ID}, e.g., https://obamawhitehouse.archives.gov/lOlHH 
will redirect to the current petition website: 
https://petitions.whitehouse.gov/petition/start-proceedings-charge-darrell-issa-ethics-violations-and-remove-him-congress/9VXmZHJm

But all the previous petitions have been removed from the current site; so once again the URL will need to be re-written to use the archive, e.g., https://petitions.obamawhitehouse.archives.gov/petition/start-proceedings-charge-darrell-issa-ethics-violations-and-remove-him-congress



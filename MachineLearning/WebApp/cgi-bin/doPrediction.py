import cgi
import sentimentAnalysis

form = cgi.FieldStorage()
review = form.getvalue('review')

pred = sentimentAnalysis.testData(review)

if pred[0] == 1:
    msg = "Positive"
else: msg = "Negative"

print("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <h1>Sentiment Analysis System</h1>
    <h2>Prediction is {} </h2>
</body>
</html>
""".format(msg))
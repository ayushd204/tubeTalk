from flask import Flask,render_template,request
from src.main import main_chain

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get",methods = ["GET","POST"])
def get_message():
    message = request.form.get("msg")
    response = main_chain.invoke(message)
    return response


if __name__ == "__main__":
    app.run(debug=True)


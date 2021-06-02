from MyProj.Api import apiBp
from MyProj import app  
app.register_blueprint(apiBp)

if __name__ == '__main__':
    app.run(port=5005,debug=True)
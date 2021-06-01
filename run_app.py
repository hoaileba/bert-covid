from MyProj.Api import apiBp
from MyProj import app  
# app = create_app()
app.register_blueprint(apiBp)
# load_dotenv('.env')

if __name__ == '__main__':
    app.run(port=5005,debug=True)
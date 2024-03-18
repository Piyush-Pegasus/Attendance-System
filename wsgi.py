from app import app,socketio

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=443, debug=True, ssl_context=('cert.pem', 'key.pem'))
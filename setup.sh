<<<<<<< HEAD
#Create non existing config folder 
mkdir ~/.streamlit
#Create config file
touch ~/.streamlit/config.toml
#Init the config
streamlit config show > ~/.streamlit/config.toml
#override the config 
mkdir -p ~/.streamlit/echo "[server]\n headless = true\n port = $PORT\n enableCORS = false\n \n" > ~/.streamlit/config.toml

echo PORT $PORT
=======
mkdir -p ~/.streamlit/echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
>>>>>>> a562f1dbfe835683cb265ae6e8095df378afce7a

mkdir -p ~/.streamlit/

echo "\
[server]\n\r\
headless = true\n\r\
enableCORS=true\n\r\
port = $PORT\n\r\
\n\r
[client]\n\r
caching = true\n\r
" > ~/.streamlit/config.toml

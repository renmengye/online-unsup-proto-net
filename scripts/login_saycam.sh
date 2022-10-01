read -p 'Email: ' uservar
read -sp 'Password: ' password
wget --post-data="email=$uservar&password=$password" --cookies=on --keep-session-cookies --save-cookies=cookie.txt "https://nyu.databrary.org/api/user/login"
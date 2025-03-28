---
title: Superman
---

# Superman

<html>
  <head>
    <title>Redirecting to Superman's documentation</title>
    <style>
      :root {
        --background-color: #ffffff;
        --text-color: #333333;
        --popup-background: #ffffff;
        --popup-shadow: rgba(0, 0, 0, 0.5);
        --button-background: #f0f0f0;
        --button-text: #333333;
        --button-hover: #e0e0e0;
        --button-border: #cccccc;
        --link-color: #0066cc;
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --background-color: #222222;
          --text-color: #e0e0e0;
          --popup-background: #333333;
          --popup-shadow: rgba(0, 0, 0, 0.7);
          --button-background: #444444;
          --button-text: #e0e0e0;
          --button-hover: #555555;
          --button-border: #666666;
          --link-color: #66b0ff;
        }
      }

      body {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.5;
        margin: 20px;
        transition: background-color 0.3s, color 0.3s;
      }

      a {
        color: var(--link-color);
        text-decoration: none;
      }

      a:hover {
        text-decoration: underline;
      }

      /* Basic styles for the popup */
      .popup {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: var(--popup-shadow);
        z-index: 999;
        justify-content: center;
        align-items: center;
      }

      .popup-content {
        background-color: var(--popup-background);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        width: 300px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: background-color 0.3s;
      }

      button {
        margin-top: 10px;
        padding: 8px 14px;
        cursor: pointer;
        background-color: var(--button-background);
        color: var(--button-text);
        border: 1px solid var(--button-border);
        border-radius: 4px;
        font-size: 14px;
        transition: background-color 0.2s, color 0.2s;
      }

      button:hover {
        background-color: var(--button-hover);
      }

      #atariRedirectBtn {
        background-color: var(--link-color);
        color: white;
        border: none;
      }

      #atariRedirectBtn:hover {
        opacity: 0.9;
      }

      label {
        display: block;
        margin: 15px 0;
        user-select: none;
      }
    </style>
  </head>
  <body>
    <p>If you are not redirected automatically, follow this <a href="https://ale.farama.org/environments/superman">link to Superman's new page</a>.</p>
    <div id="popup" class="popup">
    <div class="popup-content">
        <p>Superman's documentation has moved to <b>ale.farama.org</b></p>
        <label>
            <input type="checkbox" id="atariAutoRedirect">Enable auto-redirect next time
        </label>
        <br>
        <button id="atariRedirectBtn">Redirect to the new website</button>
        <button id="closePopupBtn">Close</button>
    </div>
    </div>
  </body>

  <script>
    // Function to get a cookie by name
    function getCookie(name) {
        console.log(`${document.cookie}`);
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) {
            return parts.pop().split(';').shift();
        }
    }

    // Function to set a cookie
    function setCookie(name, value, days) {
        const date = new Date();
        date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
        const expires = `expires=${date.toUTCString()}`;
        document.cookie = `${name}=${value}; ${expires}; path=/`;  // environments/atari/
    }

    // Show popup if the cookie doesn't exist
    window.onload = function() {
        const atariAutoRedirect = getCookie('atariAutoRedirect');
        if (atariAutoRedirect) {
            window.location.href = "https://ale.farama.org/environments/superman";
        } else {
            document.getElementById('popup').style.display = 'flex';
        }
    };

    // Close popup and handle the "Don't show again" option
    document.getElementById('closePopupBtn').addEventListener('click', function() {
        document.getElementById('popup').style.display = 'none';
    });
    document.getElementById('atariRedirectBtn').addEventListener("click", function() {
        if (document.getElementById('atariAutoRedirect').checked) {
            setCookie('atariAutoRedirect', 'true', 90);  // Set cookie to not show for 90 days
        }
        console.log("redirecting");
        window.location.href = "https://ale.farama.org/environments/superman";
    })
  </script>
</html>

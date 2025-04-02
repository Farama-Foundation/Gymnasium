---
title: Atari
firstpage:
lastpage:
---

# Atari

<html>
  <head>
    <title>Redirecting to Atari's documentation</title>
    <style>
      /* Basic styles for the popup */
      .popup {
          display: none;
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background-color: rgba(0, 0, 0, 0.5); /* Default background overlay */
          z-index: 999;
          justify-content: center;
          align-items: center;
      }
      .popup-content {
          background-color: #fff; /* Default light content background */
          padding: 20px;
          border-radius: 10px;
          text-align: center;
          width: 300px;
      }
      .dark-theme .popup {
          background-color: rgba(0, 0, 0, 0.7); /* Darker overlay for dark theme */
      }
      .dark-theme .popup-content {
          background-color: #333; /* Dark content background */
          color: #fff; /* Light text for dark theme */
      }
      button {
          margin-top: 10px;
          padding: 5px 10px;
          cursor: pointer;
      }
      /* Add spacing between checkbox and label text */
      input[type="checkbox"] {
          margin-right: 5px;
      }
    </style>
  </head>
  <body>
    <p>If you are not redirected automatically, follow this <a href="https://ale.farama.org/environments/">link to Atari's new page</a>.</p>
    <div id="popup" class="popup">
    <div class="popup-content">
        <p>Atari's documentation has moved to <b>ale.farama.org</b></p>
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

    // Function to apply theme to the popup
    function applyTheme() {
        const theme = localStorage.getItem("theme") || "auto";
        const body = document.body;

        // Remove any existing theme classes
        body.classList.remove("dark-theme", "light-theme");

        if (theme === "dark") {
            body.classList.add("dark-theme");
        } else if (theme === "light") {
            body.classList.add("light-theme");
        } else if (theme === "auto") {
            // Check system preference for dark mode
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                body.classList.add("dark-theme");
            } else {
                body.classList.add("light-theme");
            }

            // Listen for system theme changes
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
                body.classList.remove("dark-theme", "light-theme");
                body.classList.add(e.matches ? "dark-theme" : "light-theme");
            });
        }
    }

    // Show popup if the cookie doesn't exist
    window.onload = function() {
        // Apply theme first
        applyTheme();

        const atariAutoRedirect = getCookie('atariAutoRedirect');
        if (atariAutoRedirect) {
            window.location.href = "https://ale.farama.org/environments/";
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
        window.location.href = "https://ale.farama.org/environments/";
    })
  </script>
</html>

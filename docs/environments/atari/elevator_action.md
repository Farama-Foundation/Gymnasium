---
title: Elevator Action
---

# Elevator Action

<html>
  <head>
    <title>Redirecting to Elevator Action's documentation</title>
    <style>
      /* Basic styles for the popup */
      .popup {
          display: none;
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background-color: rgba(0, 0, 0, 0.5);
          z-index: 999;
          justify-content: center;
          align-items: center;
      }
      .popup-content {
          background-color: #fff;
          padding: 20px;
          border-radius: 10px;
          text-align: center;
          width: 300px;
      }
      button {
          margin-top: 10px;
          padding: 5px 10px;
          cursor: pointer;
      }
    </style>
  </head>
  <body>
    <p>If you are not redirected automatically, follow this <a href="https://ale.farama.org/environments/elevator_action">link to Elevator Action's new page</a>.</p>
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

    // Show popup if the cookie doesn't exist
    window.onload = function() {
        const atariAutoRedirect = getCookie('atariAutoRedirect');
        if (atariAutoRedirect) {
            window.location.href = "https://ale.farama.org/environments/elevator_action";
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
        window.location.href = "https://ale.farama.org/environments/elevator_action";
    })
  </script>
</html>

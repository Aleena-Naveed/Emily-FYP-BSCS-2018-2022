var express = require('express');
var router = express.Router();

var User = require('../model/User');
var passport = require('passport');
var authenticate = require('../authenticate');
const nodemailer = require('nodemailer');

const sender = 'your-email';

var transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: sender,
    pass: 'yourpassword'
  }
});

/* GET users listing. */
router.get('/', function (req, res, next) {
  res.send('respond with a resource');
});

router.post('/signup', (req, res) => {
  User.register(new User({ username: req.body.username }),
    req.body.password, (err, user) => {
      if (err) {
        console.log(err)
        res.statusCode = 500
        res.setHeader('Content-Type', 'application/json');
        res.json({ success: false });
      } else {
        if (req.body.email)
          user.email = req.body.email
        if (req.body.phone_number)
          user.contact = req.body.phone_number
        user.save((err, user) => {
          if (err) {
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json');
            res.json({ err: err.name });
            return;
          }
          passport.authenticate('local')(req, res, () => {
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json');
            res.json({
              success: true,
              user: user
            });
          });
        });
      }
    })
})

router.post('/verify', (req, res) => {
  User.findById(req.body.user._id, (err, user) => {
    if (err) {
      res.json({
        success: false,
        message: err.name
      })
    } else if (user) {
      user.contactIsVerified = true
      user.emailIsVerified = true
      user.save((err, user) => {
        if (err) {
          res.json({
            success: false,
            message: err.name
          })
        }
        else {
          res.json({
            success: true,
            user: user
          })
        }
      })
    } else {
      res.json({
        success: false,
        message: 'User not Found'
      })
    }
  })
})

router.post('/complete_profile', (req, res) => {
  User.findById(req.body.user._id, (err, user) => {
    if (err) {
      res.json({
        success: false,
        message: err.name
      })
    } else if (user) {
      user.name = req.body.fullname
      user.age = req.body.age
      user.country = req.body.country
      user.dob = req.body.dob
      user.occupation = req.body.occupation
      user.gender = req.body.gender
      user.save((err, user) => {
        if (err) {
          res.json({
            success: false,
            message: err.name
          })
        }
        else {
          var token = authenticate.getToken({ _id: user._id });
          res.json({
            success: true,
            user: user,
            token: token
          })
        }
      })
    } else {
      res.json({
        success: false,
        message: 'User not found'
      })
    }
  })
})

router.post('/login', passport.authenticate('local'), (req, res) => {
  var token = authenticate.getToken({ _id: req.user._id });
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/json');
  res.json({
    success: true,
    token: token,
    status: 'You are successfully logged in!',
    user: req.user
  });
});

router.post('/sendCode', (req, res, next) => {
  const code = parseInt(Math.random() * 10000);
  const email = req.body.email;
  var mailOptions = {
    from: sender,
    to: email,
    subject: 'Verification Code Argi',
    text: 'Verification Code is ' + String(code)
  };
  transporter.sendMail(mailOptions, function (err, info) {
    if (err) {
      console.log(err)
      res.json({ err: err, info: info })
    } else {
      res.json({ data: 'Email sent: ' + info.response, code: code, success: true });
    }
  });
})


module.exports = router;

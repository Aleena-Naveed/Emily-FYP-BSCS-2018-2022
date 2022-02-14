var express = require('express');
var router = express.Router();
var authenticate = require('../authenticate');
var User = require('../model/User');
/* GET home page. */
router.get('/', function (req, res, next) {
    res.render('index', { title: 'Express' });
});

router.post('/change_password', authenticate.verifyUser, (req, res) => {
    User.findById(req.user._id, (err, user) => {
        if (err)
            res.json({
                success: false,
                message: err.name
            })
        else if (user) {
            user.changePassword(req.body.oldpassword, req.body.newpassword, (err) => {
                if (err) {
                    if (err.name === 'IncorrectPasswordError') {
                        res.json({
                            success: false,
                            message: 'Incorrect password'
                        }); // Return error
                    } else {
                        res.json({
                            success: false,
                            message: 'Something went wrong!! Please try again after sometimes.'
                        });
                    }
                } else {
                    res.json({
                        success: true,
                        message: 'Your password has been changed successfully'
                    });
                }

            })
        } else {
            res.json({
                success: false,
                message: 'User not found'
            }); // Return error, user was not found in db
        }
    })
})

router.patch('/change_email', authenticate.verifyUser, (req, res) => {
    console.log(req.body)
    User.findById(req.user._id, (err, user) => {
        if (err) {
            res.json({
                success: false,
                message: err.name
            })
        }
        else if (user) {
            user.email = req.body.email;
            user.save((err, user) => {
                if (err)
                    res.json({
                        success: false,
                        message: err.name
                    })
                else
                    res.json({
                        success: true,
                        user: user,
                        message: 'Email Updated Successfully'
                    })
            })
        }
        else
            res.json({
                success: false,
                message: 'User not Found'
            })
    })
})


router.patch('/change_contact', authenticate.verifyUser, (req, res) => {
    User.findById(req.user._id, (err, user) => {
        if (err) {
            res.json({
                success: false,
                message: err.name
            })
        }
        else if (user) {
            user.contact = req.body.contact;
            user.save((err, user) => {
                if (err)
                    res.json({
                        success: false,
                        message: err.name
                    })
                else
                    res.json({
                        success: true,
                        user: user,
                        message: 'Email Updated Successfully'
                    })
            })
        }
        else
            res.json({
                success: false,
                message: 'User not Found'
            })
    })
})
module.exports = router;

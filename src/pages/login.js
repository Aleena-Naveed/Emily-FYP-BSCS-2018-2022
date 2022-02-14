import React from 'react';

import { Button, Grid, makeStyles, Typography, Card, TextField } from '@material-ui/core';

import logo from '../emily/Emily Logo-01.png';
import google from "../emily/google.svg";
import facebook from "../emily/facebook.svg";
import Emily from "../animations/emily_login.mp4";
import InputAdornment from "@material-ui/core/InputAdornment";
import IconButton from "@material-ui/core/IconButton";
import VisibilityOff from "@material-ui/icons/VisibilityOff";
import Visibility from "@material-ui/icons/Visibility";

import * as yup from "yup";
import { useFormik } from "formik";

import { useDispatch } from "react-redux";
import { login, setuser, settoken } from "../pages/stateSlice";
import axios from "axios";
import { useHistory } from "react-router-dom";
import Snackbar from "@material-ui/core/Snackbar";
import MuiAlert from '@material-ui/lab/Alert';

function Alert(props) {
    return <MuiAlert elevation={6} variant="filled" {...props} />;
}

const validationSchema = yup.object({
    username: yup
        .string('Enter your Username')
        .required('Username is required'),
    password: yup
        .string('Enter your password')
        .min(8, 'Password should be of minimum 8 characters length')
        .max(16, 'Password should be of maximum 16 characters length')
        .required('Password is required'),
});

const useStyles = makeStyles({
    root: {
        flexGrow: 1,
    },
    leftGrid: {
        background: "#fff",
        minHeight: "100vh",
        boxShadow: "0px 3px 6px rgba(0, 0, 0, 0.16)",
    },
    leftGridHeaderBtn: {
        borderRadius: "3%",
        boxShadow: "0px 3px 6px rgba(0, 0, 0, 0.16)",
        background: "#fff",
        color: "#1061b0",
        fontWeight: 'bold',
        overflow: 'hidden',
        right: 5,

    },
    homeText: {
        color: "#3585da",
        fontWeight: 'bold',
        padding: "5px 10px 0px 10px",
        "&:hover": {
            cursor: "pointer"
        }
    },
    loginBoxGrid: {
        justifyContent: 'center',
        alignItems: "center",
        alignContent: 'center'
    },
    loginText: {
        fontWeight: 700,
        textAlign: "center",
    },
    loginCard: {
        borderRadius: "3%",
        justifyContent: "center",
        textAling: "center",
        marginTop: 100,
        color: '#1061b0',
        padding: 30,
    },
    inputProps: {
        color: "#7ea2c4",
        fontWeight: "bold",
    },
    inputField: {
        marginBottom: 15,
    },
    contained: {
        "&:hover": {
            background: "rgba(53,133,218,0.8)"
        }
    },
    forgotPassword: {
        fontWeight: 600,
        textDecoration: 'underline',
        color: "#7ea2c4",
        marginBottom: 30,
    },
    loginBtn: {
        borderRadius: 15,
        background: "#3585da",
        boxShadow: "3px 3px 6px rgba(0, 0, 0, 0.16)",
        fontWeight: 'bold',
        color: "#fff",
    },
    loginContained: {
        "&:hover": {
            background: "rgba(53,133,218,0.8)",
        }
    },
    loginGContained: {
        "&:hover": {
            background: "rgba(220,78,65,0.8)",
        }
    },
    loginBtnText: {
        fontWeight: 'bold',
        color: "#fff",
    },
    googleBtn: {
        backgroundColor: "#dc4e41",
        fontWeight: 600,
        boxShadow: "0px 3px 6px rgba(0, 0, 0, 0.16)",
        margin: 5,
    },
    facebookBtn: {
        backgroundColor: "#1976d2",
        fontWeight: 600,
        boxShadow: "0px 3px 6px rgba(0, 0, 0, 0.16)",
        margin: 5,
    },
    btnContainer: {
        justifyContent: "space-between",
        marginTop: 25,
    },
    videoDim: {
        position: 'absolute',
        right: '0',
        bottom: '0',
        minWidth: '100vw',
        height: '100vh',
        objectFit: "cover",
        zIndex: '-1'
    }
})

export default function Login() {
    const classes = useStyles();
    const [showPassword, setShowPassword] = React.useState(false);
    const handleClickShowPassword = () => setShowPassword(!showPassword);
    const handleMouseDownPassword = () => setShowPassword(!showPassword);
    const dispatch = useDispatch();
    const history = useHistory();
    const [snackbar, setsnackbar] = React.useState({
        open: false,
        msg: "",
        type: ""
    })
    const handleClose = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }

        setsnackbar({ ...snackbar, open: false });
    };
    const formik = useFormik({
        initialValues: {
            username: '',
            password: '',
        },
        validationSchema: validationSchema,
        onSubmit: (values) => {
            axios.post("http://localhost:5000/users/login", values)
                .then(res => {
                    if (res.data.success) {
                        setsnackbar({
                            ...snackbar,
                            open: true,
                            msg: "Login Successfull",
                            type: "success"
                        })
                        setTimeout(() => {
                            dispatch(login());
                            dispatch(setuser(res.data.user));
                            dispatch(settoken(res.data.token));
                        }, 1000)
                    } else {
                        setsnackbar({
                            ...snackbar,
                            open: true,
                            msg: "Invalid Credentials",
                            type: "error"
                        })
                    }
                })
                .catch(err => {
                    setsnackbar({
                        ...snackbar,
                        open: true,
                        msg: "Invalid Credentials",
                        type: "error"
                    })
                })
        },
    });

    return (
        <Grid container className={classes.root}>
            <Grid item sm={6} className={classes.leftGrid}>
                <Grid container direction="row" justifyContent="space-between" alignItems="center" >
                    <Grid item xs={3}>
                        <img src={logo} alt="Emily Logo" style={{ width: '80px', paddingTop: '10px', paddingLeft: '20px' }} />
                    </Grid>
                    <Grid item md={5}></Grid>
                    <Grid item xs={3} md={2}>
                        <Typography
                            variant="body1"
                            className={classes.homeText}
                            onClick={() => history.push("/")}
                        >HOME</Typography>
                    </Grid>
                    <Grid item xs={6} md={2}>
                        <Button
                            fullWidth
                            variant="contained"
                            className={classes.leftGridHeaderBtn}
                            classes={{
                                contained: classes.contained
                            }} href="/signup"
                        >SIGN UP</Button>
                    </Grid>
                </Grid>
                <Grid container className={classes.loginBoxGrid}>
                    <Grid item xs={9}>
                        <Card className={classes.loginCard}>
                            <Typography variant="h4" className={classes.loginText}>LOGIN</Typography>
                            <form onSubmit={formik.handleSubmit}>
                                <TextField
                                    fullWidth
                                    label="USERNAME"
                                    name="username"
                                    InputLabelProps={{
                                        className: classes.inputProps
                                    }}
                                    inputProps={{
                                        className: classes.inputProps
                                    }}
                                    classes={{
                                        root: classes.inputField
                                    }}
                                    value={formik.values.username}
                                    onChange={formik.handleChange}
                                    error={formik.touched.username && Boolean(formik.errors.username)}
                                    helperText={formik.touched.username && formik.errors.username}
                                />
                                <TextField
                                    label="PASSWORD"
                                    name="password"
                                    fullWidth
                                    InputLabelProps={{
                                        className: classes.inputProps
                                    }}
                                    inputProps={{
                                        className: classes.inputProps,
                                    }}
                                    type={showPassword ? "text" : "password"}
                                    InputProps={{
                                        endAdornment: (
                                            <InputAdornment position="end">
                                                <IconButton
                                                    aria-label="toggle password visibility"
                                                    onClick={handleClickShowPassword}
                                                    onMouseDown={handleMouseDownPassword}
                                                >
                                                    {showPassword ? <Visibility /> : <VisibilityOff />}
                                                </IconButton>
                                            </InputAdornment>
                                        ),
                                    }} value={formik.values.password}
                                    onChange={formik.handleChange}
                                    error={formik.touched.password && Boolean(formik.errors.password)}
                                    helperText={formik.touched.password && formik.errors.password}
                                />
                                {/* <Typography variant="body1" className={classes.forgotPassword}>FORGOT PASSWORD?</Typography> */}
                                <Button fullWidth
                                    variant="contained"
                                    className={classes.loginBtn}
                                    classes={{
                                        contained: classes.loginContained,
                                        label: classes.loginBtnText,
                                    }}
                                    type="submit"
                                >
                                    LOGIN
                                </Button>
                            </form>
                        </Card>
                    </Grid>
                </Grid>
            </Grid>
            <Grid item sm={6}></Grid>
            <Grid item xs={12}>
                <video autoPlay loop muted className={classes.videoDim}>
                    <source src={Emily} type="video/mp4" />
                </video>
            </Grid>
            <Snackbar
                open={snackbar.open}
                autoHideDuration={4000}
                onClose={handleClose}
            >
                <Alert severity={snackbar.type}>
                    {snackbar.msg}
                </Alert>
            </Snackbar>
        </Grid>
    )
}
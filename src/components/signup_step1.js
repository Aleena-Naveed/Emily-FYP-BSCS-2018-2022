import React from 'react';

import Grid from '@material-ui/core/Grid';
import TextField from '@material-ui/core/TextField';
import { makeStyles } from "@material-ui/core/styles";
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import InputAdornment from "@material-ui/core/InputAdornment";
import IconButton from "@material-ui/core/IconButton";
import VisibilityOff from "@material-ui/icons/VisibilityOff";
import Visibility from "@material-ui/icons/Visibility";
import { useFormik } from 'formik';
import { basicVal } from '../pages/yup';

import axios from "axios";

import { setuser } from "../pages/stateSlice";
import { useDispatch } from "react-redux";
import { useHistory } from 'react-router';

import Snackbar from "@material-ui/core/Snackbar";
import MuiAlert from '@material-ui/lab/Alert';

function Alert(props) {
    return <MuiAlert elevation={6} variant="filled" {...props} />;
}

const useStyles = makeStyles({
    inputProps: {
        color: "#7ea2c4",
        fontWeight: "600",
        fontSize: 12,
        marginTop: 5,
    },
    Btn: {
        borderRadius: 15,
        background: "#3585da",
        boxShadow: "3px 3px 6px rgba(0, 0, 0, 0.16)",
        fontWeight: 'bold',
        color: "#fff",
        marginTop: 10,
        marginBottom: 5,
        "&:hover": {
            background: "rgba(53,133,218,0.8)",
        }
    },
    BtnText: {
        fontWeight: 'bold',
        color: "#fff",
        fontSize: 16,
    },
    typographyText: {
        fontWeight: 700,
        textAlign: "center",
        textDecoration: 'underline',
    },
    btmText: {
        textAlign: "center",
        padding: 5,
    },
    inputField: {
        fontSize: 12,
        marginTop: 5,
    },
})

export default function Step1({ setActiveStep }) {
    const [showPassword, setShowPassword] = React.useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = React.useState(false);

    const classes = useStyles();
    const handleClickShowPassword = () => setShowPassword(!showPassword);
    const handleMouseDownPassword = () => setShowPassword(!showPassword);
    const handleClickShowConfirmPassword = () => setShowConfirmPassword(!showConfirmPassword);
    const handleMouseDownConfirmPassword = () => setShowConfirmPassword(!showConfirmPassword);
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
            username: "",
            email: "",
            password: "",
            confirmpassword: "",
            phone_number: "",
        },
        validationSchema: basicVal,
        onSubmit: (values) => {
            console.log(values)
            axios.post("http://localhost:5000/users/signup", values)
                .then(res => {
                    if (res.data.success) {
                        setsnackbar({
                            ...snackbar,
                            open: true,
                            msg: "Registeration Successful",
                            type: "success"
                        })
                        setTimeout(() => {
                            dispatch(setuser(res.data.user));
                            setActiveStep(1);
                        }, 1000)
                    } else {
                        setsnackbar({
                            ...snackbar,
                            open: true,
                            msg: "Registeration Unsuccessful",
                            type: "error"
                        })
                    }
                })
                .catch(err => {
                    setsnackbar({
                        ...snackbar,
                        open: true,
                        msg: "Registeration Unsuccessful",
                        type: "error"
                    })
                })
        },
    })
    return (
        <Grid>
            <Typography variant="h5" className={classes.typographyText}>SIGNUP</Typography>
            <form onSubmit={formik.handleSubmit}>
                <TextField
                    fullWidth
                    label="USERNAME"
                    type="text"
                    name="username"
                    className={classes.inputField}
                    InputLabelProps={{
                        className: classes.inputProps
                    }}
                    inputProps={{
                        className: classes.inputProps,
                    }}
                    value={formik.values.username}
                    onChange={formik.handleChange}
                    error={formik.touched.username && Boolean(formik.errors.username)}
                    helperText={formik.touched.username && formik.errors.username} />
                <TextField
                    fullWidth
                    label="EMAIL"
                    type="email"
                    name="email"
                    className={classes.inputField}
                    InputLabelProps={{
                        className: classes.inputProps
                    }}
                    inputProps={{
                        className: classes.inputProps,
                    }}
                    value={formik.values.email}
                    onChange={formik.handleChange}
                    error={formik.touched.email && Boolean(formik.errors.email)}
                    helperText={formik.touched.email && formik.errors.email}
                />
                <TextField
                    fullWidth
                    label="0092XXXXXXXXXX"
                    type="text"
                    name="phone_number"
                    className={classes.inputField}
                    InputLabelProps={{
                        className: classes.inputProps
                    }}
                    inputProps={{
                        className: classes.inputProps,
                    }}
                    value={formik.values.phone_number}
                    onChange={formik.handleChange}
                    error={formik.touched.phone_number && Boolean(formik.errors.phone_number)}
                    helperText={formik.touched.phone_number && formik.errors.phone_number} />
                <TextField
                    fullWidth
                    label="PASSWORD"
                    name="password"
                    className={classes.inputField}
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
                    }}
                    value={formik.values.password}
                    onChange={formik.handleChange}
                    error={formik.touched.password && Boolean(formik.errors.password)}
                    helperText={formik.touched.password && formik.errors.password}
                />
                <TextField
                    fullWidth
                    label="CONFIRM PASSWORD"
                    name="confirmpassword"
                    className={classes.inputField}
                    InputLabelProps={{
                        className: classes.inputProps
                    }}
                    inputProps={{
                        className: classes.inputProps,
                    }}
                    type={showConfirmPassword ? "text" : "password"}
                    InputProps={{
                        endAdornment: (
                            <InputAdornment position="end">
                                <IconButton
                                    aria-label="toggle password visibility"
                                    onClick={handleClickShowConfirmPassword}
                                    onMouseDown={handleMouseDownConfirmPassword}
                                >
                                    {showConfirmPassword ? (
                                        <Visibility />
                                    ) : (
                                        <VisibilityOff />
                                    )}
                                </IconButton>
                            </InputAdornment>
                        ),
                    }}
                    value={formik.values.confirmpassword}
                    onChange={formik.handleChange}
                    error={formik.touched.confirmpassword && Boolean(formik.errors.confirmpassword)}
                    helperText={formik.touched.confirmpassword && formik.errors.confirmpassword}
                />
                <Button
                    fullWidth
                    variant="contained"
                    type="submit"
                    className={classes.Btn}
                    classes={{
                        contained: classes.Contained,
                        label: classes.BtnText,
                    }}
                >

                    CONTINUE
                </Button>
            </form>
            <Grid className={classes.btmText}>
                <div onClick={() => history.push('/login')} style={{ cursor: "pointer" }}>
                    <Typography
                        variant="body2"
                        display="inline"
                    >
                        HAVE AN ACCOUNT?
                    </Typography>
                    <Typography
                        variant="body2"
                        className={classes.typographyText}
                        display="inline" > SIGNIN</Typography>
                </div>
            </Grid>
            <Snackbar open={snackbar.open}
                autoHideDuration={6000}
                onClose={handleClose}
            >
                <Alert severity={snackbar.type}>
                    {snackbar.msg}
                </Alert>
            </Snackbar>
        </Grid>
    )
}
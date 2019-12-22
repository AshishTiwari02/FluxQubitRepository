package com.PageObjectModel;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.How;
import org.openqa.selenium.support.PageFactory;

public class RegistrationPage2 {

	WebDriver driver;
	
	@FindBy(how=How.LINK_TEXT,using="REGISTER" )
	WebElement RegLink;
	
	@FindBy(how=How.NAME,using="firstName")
	WebElement FirstName;
	
	@FindBy(how=How.NAME,using="lastName")
	WebElement LastName;
	
	@FindBy(how=How.NAME,using="phone")
	WebElement phone;
	
	@FindBy(how=How.NAME,using="email")
	WebElement username;
	
	@FindBy(how=How.NAME,using="password")
	WebElement password;
	
	@FindBy(how=How.NAME,using="confirmPassword")
	WebElement confirmpassword;
	
	@FindBy(how=How.NAME,using="register")
	WebElement registarBtn;
	
	RegistrationPage2(WebDriver d)
	{
		driver = d;
		PageFactory.initElements(d, this); //When POM using FindBy keywords
	}
	public void clickRegLink()
	{
		RegLink.click();
	}
	public void setFirstName(String fname)
	{
		FirstName.sendKeys(fname);
	}
	public void setLastName(String lname)
	{
		LastName.sendKeys(lname);
	}
	public void setPhone(String ph)
	{
		phone.sendKeys(ph);
	}
	public void setUserName(String uname)
	{
		username.sendKeys(uname);
	}
	public void setPassword(String pwd)
	{
		password.sendKeys(pwd);
	}
	public void setConfirmPassword(String cnfPwd)
	{
		confirmpassword.sendKeys(cnfPwd);
	}
	public void clickRegBtn(String regbtn)
	{
		registarBtn.click();
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}

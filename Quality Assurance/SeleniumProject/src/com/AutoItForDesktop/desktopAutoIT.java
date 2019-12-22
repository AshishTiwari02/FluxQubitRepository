package com.AutoItForDesktop;

import java.io.IOException;
import java.util.HashMap;

import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.remote.CapabilityType;
import org.openqa.selenium.remote.DesiredCapabilities;

public class desktopAutoIT {

	public static void main(String[] args) throws IOException {


		String driverPath = "webdriver.chrome.driver";
		String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe";
		
		System.setProperty(driverPath, driverURL);
		
		
		WebDriver driver = new ChromeDriver();
		
		WebElement button= driver.findElement(By.xpath("/html/body/select"));

		//ControlFocus("Title","Text","controlID");
		
		//Combobox = class+Instance
		//ControlFocus("File Upload","text","ComboBox");
		//ControlSetText("File Upload","text","ComboBox","file location");
		//ControlClick("File Upload","text","ComboBox");
		
		
		/*
		 * Alternative Method when click opertion not perform using AutoIT
		 * 
		 * JavascriptExecutor js = (JavascriptExecutor)driver;
		 * js.executeScript("arguments[0].click();", button);
		 */
		
		Runtime.getRuntime().exec("AutoIT executer file"+""+"File path");
	}

}
